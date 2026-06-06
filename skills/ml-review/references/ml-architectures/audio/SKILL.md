---
name: audio
description: Audio generation architectures — text-to-speech (VALL-E, XTTS, F5-TTS, Parler-TTS, Kokoro, Kyutai), music generation (MusicGen, Stable Audio Open, YuE, ACE-Step), and voice cloning / replication (zero-shot, fine-tune, voice conversion). Covers neural audio codecs (EnCodec, DAC, Mimi, SNAC) as the discrete-token foundation. Use when picking a TTS model, generating music or sound effects, cloning a voice, or choosing between autoregressive codec-LM, diffusion, and flow-matching for audio.
---

# Audio Generation

## Why This Exists

**Problem**: "Audio generation" is three different problem shapes wearing the same coat — speaking text, composing music, and replicating a target voice each have their own state-of-the-art and their own failure modes. Picking VALL-E to generate lo-fi hip-hop, or MusicGen to read a paragraph, or a diffusion TTS for a real-time voice agent are all common mistakes that waste days.

**Key insight**: Modern audio generation almost always tokenizes waveform with a **neural audio codec** (EnCodec, DAC, Mimi, SNAC) and then runs a sequence model — autoregressive transformer, diffusion, or flow-matching — over those discrete tokens. This is why TTS, music, and voice cloning share so much architecture: once audio looks like text, an LLM-style stack works on it.

**Reach for this when**: You need to choose between codec-LM TTS vs flow-matching TTS, decide whether to clone a voice zero-shot or fine-tune, pick MusicGen vs Stable Audio Open for a music-gen task, or understand why "audio LLMs" suddenly work.

---

## Architecture Diagram — The Common Pipeline

Almost every modern audio generator (TTS, music, cloning) is one of three pipelines on top of a neural codec:

```
                        TRAINING (offline, codec only)
   waveform ──→ Codec Encoder ──→ RVQ tokens ──→ Codec Decoder ──→ waveform
   (24-48 kHz)   (CNN, GRU)       (12-86 Hz,      (CNN + HiFi-GAN)  (reconstructed)
                                   N codebooks)


                          INFERENCE — three families

  ┌─ Family A: Autoregressive codec-LM (VALL-E, XTTS, MusicGen, Bark) ────────┐
  │                                                                            │
  │   text / prompt ──→ Text Enc ──┐                                           │
  │                                ├──→ AR Transformer ──→ RVQ tokens ──→ Vocoder ──→ wav
  │   ref audio (clone) ──→ Codec ─┘     (decoder-only,       (codec       (or codec
  │                                       KV-cache)           decoder)      decoder)
  └────────────────────────────────────────────────────────────────────────────┘

  ┌─ Family B: Flow-matching / Diffusion TTS (F5-TTS, E2-TTS, NaturalSpeech 3, AudioLDM 2) ─┐
  │                                                                                          │
  │   text + ref ──→ Text Enc ──→ Flow / Diffusion ──→ mel or codec features ──→ Vocoder ──→ wav
  │                               (iterative ODE/SDE,       (parallel,            (HiFi-GAN
  │                                T5/CLAP cond)             non-AR)               BigVGAN)
  └──────────────────────────────────────────────────────────────────────────────────────────┘

  ┌─ Family C: Streaming / duplex (Moshi, Kyutai TTS) ────────────────────────────────────┐
  │                                                                                        │
  │   text stream ──→ Text Enc ──→ Frame-sync Transformer ──→ Mimi tokens ──→ Mimi Dec ──→ wav
  │                                (12.5 Hz, delayed-streams,  (semantic +     (low-latency
  │                                 audio-text interleaved)     acoustic)       streaming)
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

**Voice cloning as a knob**: in Family A, the reference clip's codec tokens are *prepended* to the AR context. In Family B, the reference is fed as an extra input to the flow / diffusion conditioning. Same model architecture, different prompt construction.

---

## The Codec Foundation

Before anything else: most modern audio generators don't predict raw waveforms — they predict **codec tokens** at 12-75 Hz, then decode back to 24-48 kHz audio.

| Codec | Rate | Codebooks (RVQ) | Notable use |
|-------|------|-----------------|-------------|
| **EnCodec** (Meta) | 75 Hz @ 24 kHz | 8 | MusicGen, AudioGen, VALL-E |
| **SoundStream** (Google) | 50 Hz @ 24 kHz | 1-N | AudioLM, MusicLM |
| **DAC** (Descript) | 86 Hz @ 44.1 kHz | 9 | Higher-fidelity TTS / music |
| **Mimi** (Kyutai) | 12.5 Hz @ 24 kHz | 8 (semantic + acoustic) | Moshi, Kyutai TTS streaming |
| **SNAC** | Multi-scale RVQ | 3 levels | Small open TTS (Orpheus, etc.) |

**Residual Vector Quantization (RVQ)**: each frame is quantized in N stages where stage k+1 encodes the residual error of stage k. You get N parallel "streams" of integer tokens per frame, low vocabulary per codebook (~1024-4096), and near-transparent reconstruction. RVQ is what makes audio look like text to a transformer.

**When this matters**: if you're choosing a model, the codec sets the fidelity ceiling and the streaming latency floor. A 12.5 Hz codec (Mimi) means 12.5 tokens/sec of audio — cheap to stream; an 86 Hz DAC at 9 codebooks means 774 tokens/sec — high quality but slow to decode autoregressively.

---

# Part 1 — Text-to-Speech (TTS)

## TTS Architecture Families

| Family | Idea | Examples | Trade-off |
|--------|------|----------|-----------|
| **Autoregressive codec-LM** | Decoder-only LM over codec tokens, text as prefix | **VALL-E**, **XTTS-v2**, **Bark**, **Parler-TTS** | Best zero-shot cloning, slower, can hallucinate |
| **Non-autoregressive parallel** | Predict all frames in parallel + duration model | **FastSpeech2**, **VITS** | Fast, deterministic, lower expressivity |
| **Diffusion TTS** | Iterative denoising of latent / codec / mel features | **NaturalSpeech 2/3**, **E2-TTS** | High quality, multi-step inference |
| **Flow-matching TTS** | Continuous-time vector field, fewer NFE than diffusion | **F5-TTS**, **Voicebox**, **Matcha-TTS** | SOTA quality/speed; current default for new work |
| **Streaming / duplex** | Frame-synchronous output, low first-token latency | **Kyutai TTS**, **Moshi** | Real-time / conversational only |
| **Vocoder layer** | Mel/feature → 24 kHz waveform | **HiFi-GAN**, **BigVGAN** | Sits under any of the above |

## TTS Decision Table

| Need | Pick | Why |
|------|------|-----|
| Real-time / streaming / duplex | **Kyutai TTS, Moshi** | Frame-synchronous, <300 ms first token |
| Zero-shot cloning from 3-30 s | **F5-TTS, XTTS-v2, E2-TTS** | Flow-matching / AR with in-context speaker prompt |
| Multilingual (15+ langs) | **XTTS-v2, Parler-TTS, Bark** | XTTS = 17 langs |
| Quality-first / studio | **F5-TTS, NaturalSpeech 3** | Flow-matching SOTA on LibriSpeech |
| Tiny / on-device / CPU | **Kokoro-82M, Piper, Matcha-TTS** | <100 M params, real-time on CPU |
| Commercial-friendly weights | **Parler-TTS** (Apache-2.0), **Kokoro** (Apache-2.0) | Most "open" TTS are non-commercial — check license |
| Prompt-controllable style | **Parler-TTS** | Natural-language style descriptions |

> License gotcha: **F5-TTS** is CC-BY-NC, **XTTS-v2** is CPML (non-commercial), **Bark** is MIT. Always verify before shipping.

## TTS Inference — Kokoro-82M (small, Apache-2.0, CPU-real-time)

```python
# pip install -U kokoro soundfile
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code="a")  # 'a' = American English
generator = pipeline("The future of speech synthesis sounds like this.",
                     voice="af_heart", speed=1.0)
for i, (_, _, audio) in enumerate(generator):
    sf.write(f"out_{i}.wav", audio, 24000)
```

## TTS Inference — Parler-TTS (HuggingFace, prompt-controllable)

```python
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

description = "A warm female voice speaking calmly with clean studio audio."
prompt = "Hello, world."
desc_ids = tok(description, return_tensors="pt").input_ids.to(device)
prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
audio = model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids).cpu().numpy().squeeze()
sf.write("out.wav", audio, model.config.sampling_rate)
```

---

# Part 2 — Music & General Audio Generation

## Music Architecture Families

| Family | Idea | Representative |
|--------|------|----------------|
| **AR codec-LM** | Decoder-only transformer over EnCodec RVQ tokens; "delay pattern" interleaves codebooks | **MusicGen**, **AudioGen**, **Jukebox** |
| **Latent diffusion / flow on audio** | Diffusion or flow-matching in a VAE latent space; faster than waveform diffusion, higher fidelity than mel | **Stable Audio Open**, **AudioLDM 2** |
| **Hybrid semantic→acoustic** | Stage 1: coarse semantic tokens (w2v-BERT/MuLan). Stage 2: acoustic codec tokens | **AudioLM**, **MusicLM** |
| **Spectrogram-as-image diffusion** | Treat mel as RGB, run SD, vocode back | **Riffusion** (mostly historical) |
| **Lyrics-aware text-to-song** | Joint lyric + style conditioning; full vocals + accompaniment | **YuE**, **ACE-Step**, (closed: Suno, Udio) |

## Key Conditioning Inputs

- **Text** via frozen **T5** (MusicGen, AudioLDM) or **CLAP** joint embeddings (Stable Audio).
- **Melody** via chromagram features (MusicGen-Melody) — robust to timbre.
- **Lyrics** via character/phoneme embeddings aligned to song structure (YuE, ACE-Step).
- **Sample-rate / channel constraints** matter: MusicGen = 32 kHz mono, ≤30 s; Stable Audio Open = 44.1 kHz stereo, ≤47 s; YuE / ACE-Step target 44.1-48 kHz stereo, multi-minute.

## Music Decision Table

| Use case | Pick | Why |
|----------|------|-----|
| Text-to-music, instrumental | **MusicGen** (medium/large) or **Stable Audio Open** | MusicGen for AR control + melody cond.; SAO for stereo 44.1 kHz |
| General SFX / Foley | **Stable Audio Open**, **AudioGen** | Trained on Freesound / AudioSet |
| Vocals + lyrics (full song) | **YuE** or **ACE-Step** | Open-weight lyric-aware text-to-song |
| Continuation / inpainting | **MusicGen** (audio-prompted) or **Stable Audio Open** (audio-to-audio) | AR continues naturally; latent diffusion supports masked inpainting |
| Commercial-licensable weights | **Stable Audio Open 1.0** (Stability Community License), **ACE-Step** (Apache-2.0) | MusicGen weights are CC-BY-NC; YuE is research-only |
| Melody-conditioned | **MusicGen-Melody** | Only major open model with chromagram cond. |
| Speech + music + SFX in one | **AudioLDM 2** | Unified latent space across modalities |

## Music Inference — MusicGen via HuggingFace

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wav
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

inputs = processor(text=["lo-fi hip hop with a mellow piano"], padding=True, return_tensors="pt").to(device)
audio = model.generate(**inputs, max_new_tokens=512)  # ~10 s at 32 kHz
wav.write("out.wav", model.config.audio_encoder.sampling_rate,
          audio[0, 0].cpu().numpy())
```

---

# Part 3 — Voice Cloning / Voice Replication

## Cloning Approaches

| Approach | Idea | Examples |
|----------|------|----------|
| **Zero-shot in-context** | 3-30 s reference audio at inference; speaker info as prompt tokens or acoustic prompt — no training | **XTTS-v2**, **F5-TTS**, **CosyVoice 2**, **OpenVoice v2**, **VALL-E** |
| **Few-shot fine-tune** | LoRA / speaker-embedding fine-tune on minutes-to-hours of clean target audio; higher fidelity than zero-shot | **Tortoise-TTS** fine-tune, **StyleTTS2** LoRA, **XTTS-v2** fine-tune |
| **Speaker-embedding conditioning** | Pretrained d-vector / x-vector / **ECAPA-TDNN** / WavLM-SV embedding as global condition | YourTTS, SpeechT5, FreeVC speaker encoder |
| **Voice conversion (any-to-any)** | Map source speech to target timbre, preserving content; no transcript needed | **RVC**, **so-vits-svc** (singing), **FreeVC**, **kNN-VC** (training-free) |
| **Cross-lingual cloning** | Clone a voice and speak a different language than the reference | **XTTS-v2** (17 langs), **OpenVoice v2**, **CosyVoice 2** |

## Quality Metrics

- **Speaker similarity (SECS / SIM-O)**: cosine similarity of WavLM-SV or ECAPA-TDNN embeddings. SOTA zero-shot ≈ 0.55-0.75.
- **WER / CER**: Whisper-large-v3 on synthesized speech; target <5 % WER for English.
- **Naturalness**: **UTMOS** (predicted MOS, 0-5) for fast offline eval; subjective MOS / CMOS for ground truth; **DNSMOS** for noise/distortion.

## Cloning Decision Table

| Need | Pick |
|------|------|
| <30 s reference, one-off, English | **Zero-shot** (F5-TTS, XTTS-v2) |
| Reference language ≠ output language | **Cross-lingual zero-shot** (XTTS-v2, OpenVoice v2, CosyVoice 2) |
| ≥10 min clean audio, want max similarity / consistent prosody | **Few-shot fine-tune** (StyleTTS2 LoRA, XTTS-v2 fine-tune) |
| Real-time / streaming (<300 ms first chunk) | **CosyVoice 2 streaming** or **kNN-VC**; avoid Tortoise / VALL-E |
| No transcript — convert existing audio (singing, dubbing) | **Voice conversion** (RVC, so-vits-svc for singing, kNN-VC training-free) |
| Strict consent / legal scrutiny | Fine-tune on **explicitly licensed** audio; never zero-shot from scraped clips |

## Cloning Inference — F5-TTS (zero-shot, flow-matching)

```python
# pip install f5-tts
from f5_tts.api import F5TTS
import soundfile as sf

tts = F5TTS()  # downloads SWivid/F5-TTS from HF
wav, sr, _ = tts.infer(
    ref_file="reference_10s.wav",
    ref_text="Transcript of the reference clip.",
    gen_text="Hello, this is my cloned voice speaking new text.",
    nfe_step=32,           # quality vs. speed
    cfg_strength=2.0,
)
sf.write("out.wav", wav, sr)
```

XTTS-v2 alternative (Coqui, CPML non-commercial):

```python
from TTS.api import TTS
TTS("tts_models/multilingual/multi-dataset/xtts_v2").tts_to_file(
    text="Hello world.", speaker_wav="ref.wav", language="en", file_path="out.wav")
```

## Safety / Ethics — Required Reading

Voice cloning enables impersonation fraud and non-consensual deepfakes. Before deploying:

1. **Documented consent** from the speaker for the specific use case — generic "voice samples online" is not consent.
2. **Refuse cloning of public figures, minors, or third parties** without authorization.
3. **Watermark generated audio** (e.g. Meta **AudioSeal**, an inaudible watermark robust to common audio edits) and add a visible disclosure when releasing.
4. **Rate-limit and log** cloning APIs to make abuse traceable.

---

## See Also

- [Attention](../attention/) — KV-cache and serving patterns underlying AR codec-LMs (VALL-E, MusicGen, Parler-TTS).
- [Diffusion](../diffusion/) — DDPM / DDIM / classifier-free guidance for diffusion TTS and AudioLDM 2.
- [LLM](../llm/) — RoPE, GQA, sampling — same techniques apply to codec-LM TTS.
- [Quantization](../quantization/) — GGUF / AWQ for shipping TTS models to edge / CPU.
- [Embeddings](../embeddings/) — speaker-embedding and CLAP audio-text embedding usage.

---

## References

### Codecs
- [EnCodec (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438) — RVQ neural codec; standard MusicGen / VALL-E backbone
- [SoundStream (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312) — original streaming neural codec
- [DAC — Descript Audio Codec (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546) — 44.1 kHz near-transparent
- [SNAC](https://github.com/hubertsiuzdak/snac) — multi-scale RVQ for small TTS
- [audiocraft repo (EnCodec + MusicGen + AudioGen)](https://github.com/facebookresearch/audiocraft)
- [descript-audio-codec repo](https://github.com/descriptinc/descript-audio-codec)
- [encodec repo](https://github.com/facebookresearch/encodec)

### TTS
- [VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111) — neural codec language model, zero-shot cloning
- [XTTS (Casanova et al., 2024)](https://arxiv.org/abs/2406.04904) — multilingual zero-shot, [coqui-ai/TTS](https://github.com/coqui-ai/TTS), [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
- [Parler-TTS (Lyth & King, 2024)](https://arxiv.org/abs/2402.01912) — natural-language style control, [huggingface/parler-tts](https://github.com/huggingface/parler-tts), [parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1)
- [Bark](https://github.com/suno-ai/bark) — codec-LM TTS with sound effects, [suno/bark](https://huggingface.co/suno/bark)
- [F5-TTS (Chen et al., 2024)](https://arxiv.org/abs/2410.06885) — flow-matching, [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS), [HF model](https://huggingface.co/SWivid/F5-TTS)
- [E2-TTS (Eskimez et al., 2024)](https://arxiv.org/abs/2406.18009) — embarrassingly easy fully non-autoregressive
- [Voicebox (Le et al., 2023)](https://arxiv.org/abs/2306.15687) — flow-matching multi-task speech
- [Matcha-TTS (Mehta et al., 2023)](https://arxiv.org/abs/2309.03199) — conditional flow matching, [repo](https://github.com/shivammehta25/Matcha-TTS)
- [NaturalSpeech 3 (Ju et al., 2024)](https://arxiv.org/abs/2403.03100) — factorized codec + diffusion
- [FastSpeech 2 (Ren et al., 2020)](https://arxiv.org/abs/2006.04558) — non-autoregressive baseline
- [VITS (Kim et al., 2021)](https://arxiv.org/abs/2106.06103) — end-to-end with flow + GAN, [repo](https://github.com/jaywalnut310/vits)
- [Moshi / Mimi (Défossez et al., 2024)](https://arxiv.org/abs/2410.00037) — full-duplex speech LM, [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- [Kyutai delayed-streams-modeling (TTS)](https://github.com/kyutai-labs/delayed-streams-modeling), [kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646) — vocoder, [repo](https://github.com/jik876/hifi-gan)
- [BigVGAN (Lee et al., 2022)](https://arxiv.org/abs/2206.04658) — universal vocoder, [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — small Apache-2.0 TTS

### Music & General Audio
- [MusicGen (Copet et al., 2023)](https://arxiv.org/abs/2306.05284) — AR codec-LM with delay pattern, [musicgen-large](https://huggingface.co/facebook/musicgen-large), [musicgen-melody](https://huggingface.co/facebook/musicgen-melody)
- [AudioGen (Kreuk et al., 2022)](https://arxiv.org/abs/2209.15352) — text-to-SFX, [audiogen-medium](https://huggingface.co/facebook/audiogen-medium)
- [AudioLM (Borsos et al., 2022)](https://arxiv.org/abs/2209.03143) — semantic + acoustic two-stage
- [MusicLM (Agostinelli et al., 2023)](https://arxiv.org/abs/2301.11325) — joint text-music embedding (MuLan)
- [Jukebox (Dhariwal et al., 2020)](https://arxiv.org/abs/2005.00341) — VQ-VAE + AR transformer
- [Stable Audio Open (Evans et al., 2024)](https://arxiv.org/abs/2407.14358) — latent diffusion, 44.1 kHz stereo, [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), [HF model](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [AudioLDM 2 (Liu et al., 2023)](https://arxiv.org/abs/2308.05734) — unified speech / music / SFX, [repo](https://github.com/haoheliu/AudioLDM2), [HF model](https://huggingface.co/cvssp/audioldm2)
- [Riffusion](https://github.com/riffusion/riffusion) — spectrogram-as-image SD, [HF model](https://huggingface.co/riffusion/riffusion-model-v1)
- [YuE (Yuan et al., 2025)](https://arxiv.org/abs/2503.08638) — open long-form text-to-song with vocals, [repo](https://github.com/multimodal-art-projection/YuE), [HF model](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot)
- [ACE-Step Technical Report (2025)](https://arxiv.org/abs/2506.00045) — Apache-2.0 text-to-song, [repo](https://github.com/ace-step/ACE-Step), [HF model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)

### Voice Cloning & Conversion
- [OpenVoice (Qin et al., 2023)](https://arxiv.org/abs/2312.01479) — tone-color converter, cross-lingual, [repo](https://github.com/myshell-ai/OpenVoice)
- [CosyVoice 2 (Du et al., 2024)](https://arxiv.org/abs/2412.10117) — streaming LLM + flow matching, [repo](https://github.com/FunAudioLLM/CosyVoice)
- [StyleTTS 2 (Li et al., 2023)](https://arxiv.org/abs/2305.07243) — diffusion-style + adversarial, [repo](https://github.com/yl4579/StyleTTS2)
- [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts) — early high-quality codec-LM TTS
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — retrieval + HuBERT voice conversion
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) — singing voice conversion
- [FreeVC (Li et al., 2022)](https://arxiv.org/abs/2210.15418) — WavLM bottleneck + VITS, [repo](https://github.com/OlaWod/FreeVC)
- [kNN-VC (Baas et al., 2023)](https://arxiv.org/abs/2305.18975) — training-free nearest-neighbor on WavLM features, [repo](https://github.com/bshall/knn-vc)

### Evaluation & Safety
- [ECAPA-TDNN (Desplanques et al., 2020)](https://arxiv.org/abs/2005.07143) — speaker-verification embedding
- [WavLM-SV checkpoint](https://huggingface.co/microsoft/wavlm-base-plus-sv) — speaker similarity backbone
- [UTMOS (Saeki et al., 2022)](https://arxiv.org/abs/2204.02152) — predicted MOS, [repo](https://github.com/sarulab-speech/UTMOS22)
- [AudioSeal (San Roman et al., 2024)](https://arxiv.org/abs/2401.17264) — proactive localized watermarking for voice cloning detection, [repo](https://github.com/facebookresearch/audioseal)
