#include "AudioPluginInterface.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

static constexpr int MAX_CHANNELS = 8;

// Partitioned FFT convolution (uniform partitioned overlap-add).
// This is dramatically faster than time-domain convolution for long IRs.

static inline int NextPow2(int v)
{
    v = std::max(1, v);
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static inline int Clamp(int x, int lo, int hi)
{
    return std::min(std::max(x, lo), hi);
}

struct FFT
{
    int N = 0;
    std::vector<int> bitrev;
    std::vector<std::complex<float>> roots; // forward roots, size N/2

    void Init(int n)
    {
        N = n;
        bitrev.assign((size_t)N, 0);
        roots.assign((size_t)N / 2, { 0.0f, 0.0f });

        int logN = 0;
        while ((1 << logN) < N)
            ++logN;

        for (int i = 0; i < N; ++i)
        {
            int x = i;
            int r = 0;
            for (int b = 0; b < logN; ++b)
            {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            bitrev[(size_t)i] = r;
        }

        constexpr float kPi = 3.14159265358979323846f;
        for (int k = 0; k < N / 2; ++k)
        {
            float phase = -2.0f * kPi * (float)k / (float)N;
            roots[(size_t)k] = { std::cos(phase), std::sin(phase) };
        }
    }

    void Transform(std::complex<float>* a, bool inverse) const
    {
        // bit-reversal permutation
        for (int i = 0; i < N; ++i)
        {
            const int j = bitrev[(size_t)i];
            if (i < j)
                std::swap(a[i], a[j]);
        }

        for (int len = 2; len <= N; len <<= 1)
        {
            const int half = len >> 1;
            const int step = N / len;
            for (int i = 0; i < N; i += len)
            {
                for (int j = 0; j < half; ++j)
                {
                    const std::complex<float> w = inverse
                        ? std::conj(roots[(size_t)j * (size_t)step])
                        : roots[(size_t)j * (size_t)step];
                    const std::complex<float> u = a[i + j];
                    const std::complex<float> v = a[i + j + half] * w;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }

        if (inverse)
        {
            const float invN = 1.0f / (float)N;
            for (int i = 0; i < N; ++i)
                a[i] *= invN;
        }
    }
};

struct EffectData
{
    int samplerate = 48000;
    int blockFrames = 1024;       // Unity DSP buffer size
    int partitionSize = 1024;     // next pow2 of blockFrames
    int fftSize = 2048;           // 2 * partitionSize
    int maxIR = 48000;            // max IR frames supported
    int maxPartitions = 1;        // ceil(maxIR / partitionSize)

    FFT fft;

    // Input spectra history per channel: [ch][partitionSlot][fftBin]
    std::vector<std::complex<float>> xHistory; // size = MAX_CHANNELS * maxPartitions * fftSize
    int xWriteIndex[MAX_CHANNELS]{};

    // IR in frequency domain (flattened): [partition][fftBin]
    std::vector<std::complex<float>> H_A; // size = maxPartitions * fftSize
    std::vector<std::complex<float>> H_B; // size = maxPartitions * fftSize
    int partsA = 0;
    int partsB = 0;

    // Overlap buffers per channel per IR
    std::vector<float> overlapA; // size = MAX_CHANNELS * partitionSize
    std::vector<float> overlapB; // size = MAX_CHANNELS * partitionSize

    // Crossfade
    float alpha = 0.0f;              // current alpha (0..1)
    int fadeSamplesRemaining = 0;
    int fadeSamplesTotal = 0;

    // Scratch buffers (reused per channel, no allocations on audio thread)
    std::vector<std::complex<float>> fftBuf;   // size = fftSize
    std::vector<std::complex<float>> accumA;   // size = fftSize
    std::vector<std::complex<float>> accumB;   // size = fftSize
    std::vector<float> outA;                  // size = blockFrames
    std::vector<float> outB;                  // size = blockFrames
    std::vector<float> alphaPerSample;        // size = blockFrames

    // Staging (written from non-audio thread). Sequence guard: odd = writing.
    std::vector<std::complex<float>> stagingH; // size = maxPartitions * fftSize
    std::atomic<std::uint32_t> stagingSeq{ 0 };
    std::atomic<int> stagingParts{ 0 };
    std::atomic<int> stagingLen{ 0 };
    std::uint32_t lastAppliedStagingSeq = 0; // audio-thread only
};

static std::atomic<EffectData*> g_singleInstance{ nullptr };

static inline void ClearEngineState(EffectData* d)
{
    std::fill(d->xHistory.begin(), d->xHistory.end(), std::complex<float>(0.0f, 0.0f));
    std::fill(d->overlapA.begin(), d->overlapA.end(), 0.0f);
    std::fill(d->overlapB.begin(), d->overlapB.end(), 0.0f);
    for (int ch = 0; ch < MAX_CHANNELS; ++ch)
        d->xWriteIndex[ch] = 0;
}

static void BuildIRFrequencyDomain(
    const FFT& fft,
    int partitionSize,
    int fftSize,
    int maxPartitions,
    const float* ir,
    int irLen,
    std::complex<float>* outH,
    int* outParts)
{
    const int clampedLen = Clamp(irLen, 0, partitionSize * maxPartitions);
    const int parts = (clampedLen <= 0) ? 0 : (int)((clampedLen + partitionSize - 1) / partitionSize);
    *outParts = parts;

    // Zero entire H (so unused partitions are neutral)
    std::fill(outH, outH + (size_t)maxPartitions * (size_t)fftSize, std::complex<float>(0.0f, 0.0f));

    if (parts == 0)
        return;

    std::vector<std::complex<float>> tmp((size_t)fftSize);
    for (int p = 0; p < parts; ++p)
    {
        std::fill(tmp.begin(), tmp.end(), std::complex<float>(0.0f, 0.0f));
        const int base = p * partitionSize;
        const int copyLen = std::min(partitionSize, clampedLen - base);
        for (int i = 0; i < copyLen; ++i)
            tmp[(size_t)i] = { ir[base + i], 0.0f };

        fft.Transform(tmp.data(), false);
        std::memcpy(outH + (size_t)p * (size_t)fftSize, tmp.data(), sizeof(std::complex<float>) * (size_t)fftSize);
    }
}

// =========================
// Exported API (C# P/Invoke)
// =========================
extern "C" UNITY_AUDIODSP_EXPORT_API int AUDIO_CALLING_CONVENTION DynamicConvolution_SetIR(const float* ir, int length)
{
    EffectData* d = g_singleInstance.load(std::memory_order_acquire);
    if (!d || !ir || length <= 0)
        return 0;

    const int clampedLen = Clamp(length, 0, d->maxIR);
    if (clampedLen <= 0)
        return 0;

    // Precompute staging H on the calling thread (avoids heavy work on audio thread).
    std::vector<std::complex<float>> localH((size_t)d->maxPartitions * (size_t)d->fftSize);
    int localParts = 0;
    BuildIRFrequencyDomain(d->fft, d->partitionSize, d->fftSize, d->maxPartitions, ir, clampedLen, localH.data(), &localParts);

    // Publish staging update guarded by sequence (odd/even).
    const std::uint32_t seq0 = d->stagingSeq.load(std::memory_order_relaxed);
    d->stagingSeq.store(seq0 + 1u, std::memory_order_release);

    if ((int)d->stagingH.size() != d->maxPartitions * d->fftSize)
        d->stagingH.assign((size_t)d->maxPartitions * (size_t)d->fftSize, std::complex<float>(0.0f, 0.0f));

    std::memcpy(d->stagingH.data(), localH.data(), sizeof(std::complex<float>) * (size_t)d->maxPartitions * (size_t)d->fftSize);
    d->stagingParts.store(localParts, std::memory_order_release);
    d->stagingLen.store(clampedLen, std::memory_order_release);

    d->stagingSeq.store(seq0 + 2u, std::memory_order_release);
    return 1;
}

static inline void ApplyPendingIRUpdateIfAny(EffectData* d)
{
    const std::uint32_t seq1 = d->stagingSeq.load(std::memory_order_acquire);
    if (seq1 & 1u)
        return;
    if (seq1 == d->lastAppliedStagingSeq)
        return;

    const int parts = d->stagingParts.load(std::memory_order_acquire);
    if (parts <= 0)
        return;

    if ((int)d->stagingH.size() != d->maxPartitions * d->fftSize)
        return;

    const std::uint32_t seq2 = d->stagingSeq.load(std::memory_order_acquire);
    if (seq2 != seq1 || (seq2 & 1u))
        return;

    d->lastAppliedStagingSeq = seq2;

    if (d->partsA == 0)
    {
        std::memcpy(d->H_A.data(), d->stagingH.data(), sizeof(std::complex<float>) * (size_t)d->maxPartitions * (size_t)d->fftSize);
        d->partsA = parts;
        d->partsB = 0;
        d->alpha = 0.0f;
        d->fadeSamplesRemaining = 0;
        std::fill(d->overlapA.begin(), d->overlapA.end(), 0.0f);
        std::fill(d->overlapB.begin(), d->overlapB.end(), 0.0f);
    }
    else
    {
        std::memcpy(d->H_B.data(), d->stagingH.data(), sizeof(std::complex<float>) * (size_t)d->maxPartitions * (size_t)d->fftSize);
        d->partsB = parts;
        d->alpha = 0.0f;
        d->fadeSamplesRemaining = d->fadeSamplesTotal;
        std::fill(d->overlapB.begin(), d->overlapB.end(), 0.0f);
    }
}

static inline const std::complex<float>* XSlot(const EffectData* d, int ch, int slot)
{
    return d->xHistory.data() + ((size_t)ch * (size_t)d->maxPartitions + (size_t)slot) * (size_t)d->fftSize;
}

static inline std::complex<float>* XSlot(EffectData* d, int ch, int slot)
{
    return d->xHistory.data() + ((size_t)ch * (size_t)d->maxPartitions + (size_t)slot) * (size_t)d->fftSize;
}

static inline float* OverlapSlot(std::vector<float>& overlap, int partitionSize, int ch)
{
    return overlap.data() + (size_t)ch * (size_t)partitionSize;
}

// =========================
// Unity callbacks
// =========================
UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK CreateCallback(UnityAudioEffectState* state)
{
    auto* d = new EffectData();

    d->samplerate = (state && state->samplerate > 0) ? (int)state->samplerate : 48000;
    d->blockFrames = (state && state->dspbuffersize > 0) ? (int)state->dspbuffersize : 1024;
    d->partitionSize = NextPow2(d->blockFrames);
    d->fftSize = d->partitionSize * 2;

    // Support up to 1 second IR at current samplerate (cap).
    d->maxIR = Clamp(d->samplerate, 1, 96000);
    d->maxPartitions = std::max(1, (d->maxIR + d->partitionSize - 1) / d->partitionSize);

    d->fft.Init(d->fftSize);

    d->xHistory.assign((size_t)MAX_CHANNELS * (size_t)d->maxPartitions * (size_t)d->fftSize, { 0.0f, 0.0f });
    d->H_A.assign((size_t)d->maxPartitions * (size_t)d->fftSize, { 0.0f, 0.0f });
    d->H_B.assign((size_t)d->maxPartitions * (size_t)d->fftSize, { 0.0f, 0.0f });

    d->overlapA.assign((size_t)MAX_CHANNELS * (size_t)d->partitionSize, 0.0f);
    d->overlapB.assign((size_t)MAX_CHANNELS * (size_t)d->partitionSize, 0.0f);

    d->fftBuf.assign((size_t)d->fftSize, { 0.0f, 0.0f });
    d->accumA.assign((size_t)d->fftSize, { 0.0f, 0.0f });
    d->accumB.assign((size_t)d->fftSize, { 0.0f, 0.0f });
    d->outA.assign((size_t)d->blockFrames, 0.0f);
    d->outB.assign((size_t)d->blockFrames, 0.0f);
    d->alphaPerSample.assign((size_t)d->blockFrames, 0.0f);
    d->stagingH.assign((size_t)d->maxPartitions * (size_t)d->fftSize, { 0.0f, 0.0f });

    d->fadeSamplesTotal = std::max(1, (int)(0.05f * (float)d->samplerate)); // 50ms
    d->fadeSamplesRemaining = 0;
    d->alpha = 0.0f;

    ClearEngineState(d);

    state->effectdata = d;
    g_singleInstance.store(d, std::memory_order_release);
    return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK ReleaseCallback(UnityAudioEffectState* state)
{
    auto* d = (EffectData*)state->effectdata;
    if (g_singleInstance.load(std::memory_order_acquire) == d)
        g_singleInstance.store(nullptr, std::memory_order_release);
    delete d;
    return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK ResetCallback(UnityAudioEffectState* state)
{
    auto* d = (EffectData*)state->effectdata;
    if (!d)
        return UNITY_AUDIODSP_OK;

    ClearEngineState(d);
    d->alpha = 0.0f;
    d->fadeSamplesRemaining = 0;
    return UNITY_AUDIODSP_OK;
}

UNITY_AUDIODSP_RESULT UNITY_AUDIODSP_CALLBACK ProcessCallback(
    UnityAudioEffectState* state,
    float* inbuffer,
    float* outbuffer,
    unsigned int length,
    int inchannels,
    int outchannels)
{
    auto* d = (EffectData*)state->effectdata;
    if (!d)
        return UNITY_AUDIODSP_OK;

    // This FFT engine assumes fixed buffer size. Unity mixer effects typically satisfy this.
    if ((int)length != d->blockFrames)
    {
        const int copyCh = std::min(inchannels, outchannels);
        if (copyCh > 0)
            std::memcpy(outbuffer, inbuffer, sizeof(float) * (size_t)length * (size_t)copyCh);
        if (outchannels > copyCh)
        {
            for (unsigned int n = 0; n < length; ++n)
                for (int ch = copyCh; ch < outchannels; ++ch)
                    outbuffer[n * outchannels + ch] = 0.0f;
        }
        return UNITY_AUDIODSP_OK;
    }

    ApplyPendingIRUpdateIfAny(d);

    if (d->partsA <= 0)
    {
        const int copyCh = std::min(inchannels, outchannels);
        if (copyCh > 0)
            std::memcpy(outbuffer, inbuffer, sizeof(float) * (size_t)length * (size_t)copyCh);
        if (outchannels > copyCh)
        {
            for (unsigned int n = 0; n < length; ++n)
                for (int ch = copyCh; ch < outchannels; ++ch)
                    outbuffer[n * outchannels + ch] = 0.0f;
        }
        return UNITY_AUDIODSP_OK;
    }

    const bool crossfading = (d->fadeSamplesRemaining > 0) && (d->partsB > 0);
    const float alphaStep = 1.0f / (float)std::max(1, d->fadeSamplesTotal);

    // Build alpha-per-sample ramp for this block (shared across channels)
    if (crossfading)
    {
        const int fadeCount = std::min(d->fadeSamplesRemaining, d->blockFrames);
        for (int i = 0; i < fadeCount; ++i)
            d->alphaPerSample[(size_t)i] = std::min(1.0f, d->alpha + alphaStep * (float)i);
        for (int i = fadeCount; i < d->blockFrames; ++i)
            d->alphaPerSample[(size_t)i] = 1.0f;

        d->alpha = std::min(1.0f, d->alpha + alphaStep * (float)fadeCount);
        d->fadeSamplesRemaining -= fadeCount;
    }
    else
    {
        std::fill(d->alphaPerSample.begin(), d->alphaPerSample.end(), 0.0f);
        d->alpha = 0.0f;
        d->fadeSamplesRemaining = 0;
    }

    const int maxCh = std::min(outchannels, MAX_CHANNELS);
    const int fftSize = d->fftSize;

    for (int ch = 0; ch < outchannels; ++ch)
    {
        if (ch >= maxCh)
        {
            // Pass-through for unsupported channel counts.
            for (int n = 0; n < d->blockFrames; ++n)
                outbuffer[n * outchannels + ch] = (ch < inchannels) ? inbuffer[n * inchannels + ch] : 0.0f;
            continue;
        }

        // Forward FFT of input block (zero padded to partitionSize, then to fftSize)
        for (int n = 0; n < d->partitionSize; ++n)
        {
            float s = (n < d->blockFrames && ch < inchannels) ? inbuffer[n * inchannels + ch] : 0.0f;
            d->fftBuf[(size_t)n] = { s, 0.0f };
        }
        for (int n = d->partitionSize; n < fftSize; ++n)
            d->fftBuf[(size_t)n] = { 0.0f, 0.0f };

        d->fft.Transform(d->fftBuf.data(), false);

        // Store input spectrum into ring buffer
        const int writeSlot = d->xWriteIndex[ch];
        std::memcpy(XSlot(d, ch, writeSlot), d->fftBuf.data(), sizeof(std::complex<float>) * (size_t)fftSize);

        // Accumulate A
        std::fill(d->accumA.begin(), d->accumA.end(), std::complex<float>(0.0f, 0.0f));
        for (int p = 0; p < d->partsA; ++p)
        {
            const int slot = (writeSlot - p + d->maxPartitions) % d->maxPartitions;
            const std::complex<float>* X = XSlot(d, ch, slot);
            const std::complex<float>* H = d->H_A.data() + (size_t)p * (size_t)fftSize;
            for (int k = 0; k < fftSize; ++k)
                d->accumA[(size_t)k] += H[(size_t)k] * X[(size_t)k];
        }

        // IFFT A
        d->fft.Transform(d->accumA.data(), true);
        float* overlapA = OverlapSlot(d->overlapA, d->partitionSize, ch);
        for (int n = 0; n < d->blockFrames; ++n)
            d->outA[(size_t)n] = d->accumA[(size_t)n].real() + overlapA[(size_t)n];
        for (int n = 0; n < d->partitionSize; ++n)
            overlapA[(size_t)n] = d->accumA[(size_t)(n + d->partitionSize)].real();

        if (crossfading)
        {
            // Accumulate B
            std::fill(d->accumB.begin(), d->accumB.end(), std::complex<float>(0.0f, 0.0f));
            for (int p = 0; p < d->partsB; ++p)
            {
                const int slot = (writeSlot - p + d->maxPartitions) % d->maxPartitions;
                const std::complex<float>* X = XSlot(d, ch, slot);
                const std::complex<float>* H = d->H_B.data() + (size_t)p * (size_t)fftSize;
                for (int k = 0; k < fftSize; ++k)
                    d->accumB[(size_t)k] += H[(size_t)k] * X[(size_t)k];
            }

            d->fft.Transform(d->accumB.data(), true);
            float* overlapB = OverlapSlot(d->overlapB, d->partitionSize, ch);
            for (int n = 0; n < d->blockFrames; ++n)
                d->outB[(size_t)n] = d->accumB[(size_t)n].real() + overlapB[(size_t)n];
            for (int n = 0; n < d->partitionSize; ++n)
                overlapB[(size_t)n] = d->accumB[(size_t)(n + d->partitionSize)].real();

            // Mix with alpha ramp
            for (int n = 0; n < d->blockFrames; ++n)
            {
                const float a = d->alphaPerSample[(size_t)n];
                outbuffer[n * outchannels + ch] = (1.0f - a) * d->outA[(size_t)n] + a * d->outB[(size_t)n];
            }
        }
        else
        {
            for (int n = 0; n < d->blockFrames; ++n)
                outbuffer[n * outchannels + ch] = d->outA[(size_t)n];
        }

        // Advance ring write index
        d->xWriteIndex[ch] = (writeSlot + 1) % d->maxPartitions;
    }

    // If crossfade completed this block, promote B->A.
    if (d->fadeSamplesRemaining == 0 && d->partsB > 0)
    {
        d->H_A.swap(d->H_B);
        d->partsA = d->partsB;
        d->partsB = 0;
        d->overlapA.swap(d->overlapB);
        std::fill(d->overlapB.begin(), d->overlapB.end(), 0.0f);
        d->alpha = 0.0f;
        d->fadeSamplesRemaining = 0;
    }

    return UNITY_AUDIODSP_OK;
}

// =========================
// Plugin definition
// =========================
extern "C" UNITY_AUDIODSP_EXPORT_API int AUDIO_CALLING_CONVENTION UnityGetAudioEffectDefinitions(UnityAudioEffectDefinition*** definitionptr)
{
    static UnityAudioEffectDefinition definition;
    std::memset(&definition, 0, sizeof(definition));

    definition.structsize = sizeof(UnityAudioEffectDefinition);
    definition.paramstructsize = sizeof(UnityAudioParameterDefinition);
    definition.apiversion = UNITY_AUDIO_PLUGIN_API_VERSION;
    definition.pluginversion = 0x00020000; // 2.0.0
    definition.channels = 0;

    strncpy_s(definition.name, "Dynamic Convolution", sizeof(definition.name));

    definition.create = CreateCallback;
    definition.release = ReleaseCallback;
    definition.reset = ResetCallback;
    definition.process = ProcessCallback;

    definition.numparameters = 0;
    definition.paramdefs = nullptr;
    definition.setfloatparameter = nullptr;
    definition.getfloatparameter = nullptr;
    definition.getfloatbuffer = nullptr;

    static UnityAudioEffectDefinition* defs[] = { &definition };
    *definitionptr = defs;
    return 1;
}