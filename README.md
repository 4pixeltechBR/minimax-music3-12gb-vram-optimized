# 🎵 MiniMax-Music 3.0: 12GB VRAM Optimization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![VRAM: 12GB Tested](https://img.shields.io/badge/VRAM-12GB%20Target-success.svg)](#benchmarks)
[![Portuguese Docs](https://img.shields.io/badge/Documentação-Português%20(PT--BR)-green.svg)](README_PT.md)

> **High-performance, production-grade optimization of MiniMax-Music 3.0 for consumer 12GB VRAM GPUs (NVIDIA RTX 3060, 4070).**  
> Reduces full-song generation time (~3m35s audio) from **8.5 hours down to ~24 minutes** — an acceleration of **over 19x (~95% reduction in wait time)** with zero PCIe paging and pristine vocal fidelity.

---

## ⚡ The Breakthrough: Why Standard Pipelines Fail

MiniMax-Music 3.0 is a state-of-the-art dual-stage musical generation pipeline pairing an **8-billion parameter Language Model (Qwen-8B)** for semantic audio tokens with a **2.4-billion parameter Diffusion Transformer (DiT)** and an acoustic Vocoder.

Running this stock pipeline on a 12GB VRAM GPU traditionally leads to severe bottlenecks:
1. **The Disk/RAM Offload Trap:** Standard implementations continuously swap layers to system RAM or SSD, crawling at **~0.18 tokens/sec** (~8 hours and 25 minutes per song).
2. **The INT8 Cache Degradation Trap:** Quantizing the KV-Cache to INT8 causes severe vocal muffling and loss of transient articulation during the first 20 seconds ("anesthetized vocal attack"), while still spilling over into Windows Shared GPU Memory via PCIe.

### 🛠️ Our Architectural Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SEMANTIC GENERATION (Peak VRAM: ~8.6 GB / Reserved: ~9.8 GB)       │
│                                                                             │
│  [Prompt + Lyrics] ──> [Qwen-8B in 4-bit NF4] ──> [Native BF16 KV Cache]    │
│                            (~4.4 GB in GDDR6)       (Crystal-clear attack)  │
│                                      │                                      │
│                        Generates ~5,375 Tokens (4.87 tok/s)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         [TWO-PHASE VRAM TRANSITION]
                     (Unload Qwen & RVQ -> Free ~8.6 GB)
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ PHASE 2: ACOUSTIC DIFFUSION & MASTER (Peak VRAM: ~5.0 GB)                   │
│                                                                             │
│  [Semantic Tokens] ──> [DiT 2.4B in Direct CUDA] ──> [Vocoder in CUDA]      │
│                              (6.5 minutes)                 (14 seconds)     │
│                                      │                                      │
│                [Smart Studio Cosine Fade-Out (Last 5s)]                     │
│                                      │                                      │
│                Result: Master Audio (44.1 kHz Stereo WAV)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **In-Memory 4-bit NF4 Quantization (BitsAndBytes):** Recursively converts the 253 linear layers of the Qwen-8B backbone to `BitsAndBytesLinear4bit` (NormalFloat4), reducing weights from 8.76 GB to ~4.40 GB in VRAM.
2. **Native BF16 KV Cache:** Preserves attention keys and values in pure uncompressed `bfloat16`, eliminating early-frame noise and ensuring crisp, energetic vocal articulation from the very first syllable.
3. **Two-Phase VRAM Swapping:** Runs the Language Model and RVQ in GDDR6, completely flushes them from GPU memory upon semantic completion, and loads the 2.4B DiT directly into CUDA without layer-by-layer offloading.
4. **Smart Cosine Fade-Out Engine:** Automatically analyzes the tail energy of the track and applies an analog-modeled cosine attenuation curve over the final 5 seconds if needed, avoiding abrupt endings.

---

## 📊 Benchmarks: 8 Master Tracks

All songs are complete studio compositions of **up to 3 minutes and 35 seconds (215 seconds)** generated on a single **NVIDIA GeForce RTX 3060 (12GB GDDR6)**:

| # | Track Title | Genre | Audio Duration | Production Time | Semantic Rate | Peak VRAM | Master File |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | **Shattered Glass Inside** | Nu-Metal / Melodic Alt-Rock | 3m 35s | 25.69 min | 4.80 tok/s | 8.99 GB | 36.21 MB WAV |
| **2** | **Shadows On The Dancefloor** | 80s Funk-Pop / Dance-Rock | 3m 28s | 24.65 min | 4.87 tok/s | 8.99 GB | 35.03 MB WAV |
| **3** | **Wharf Street Lanterns** | Roots Rock / British Blues | 3m 34s | 25.52 min | 4.86 tok/s | 8.94 GB | 35.97 MB WAV |
| **4** | **Crown of Thorns & Snow** | Symphonic Gothic Rock | 3m 35s | 24.37 min | 5.10 tok/s | 8.58 GB | 36.21 MB WAV |
| **5** | **Iron In The Blood** | 80s Thrash Metal / Heavy Metal | 3m 35s | 25.06 min | 4.94 tok/s | 8.58 GB | 36.21 MB WAV |
| **6** | **Antes Que O Dia Amanheça** | Pop Rock / Post-Punk Revival | 3m 35s | 25.90 min | 4.74 tok/s | 8.62 GB | 36.21 MB WAV |
| **7** | **One Breath Left** | Cinematic Hip-Hop / Rock-Rap | 2m 55s | 20.32 min | 4.95 tok/s | 8.63 GB | 29.36 MB WAV |
| **8** | **Circus of the Broken Clocks** | Avant-Garde Metal / Alt Nu-Metal | 3m 35s | 26.02 min | 4.72 tok/s | 8.62 GB | 36.21 MB WAV |

*Detailed forensic telemetry and acoustic measurements are documented in [BENCHMARKS.md](BENCHMARKS.md).*

---

## 🚀 Quickstart Guide

### 1. Requirements
* OS: Windows 10/11 or Linux
* Python: 3.10 or 3.11
* GPU: NVIDIA GPU with at least 12GB VRAM (RTX 3060, 4070, A4000, 3080 12GB)
* RAM: 32GB recommended
* CUDA: 12.1+ / PyTorch 2.1+

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/4pixeltechBR/minimax-music3-12gb-vram-optimized.git
cd minimax-music3-12gb-vram-optimized

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate a Single Track

```bash
python generate_music.py \
  --prompt "Basic Attributes: bpm is 120. key is A, and scale is minor. 80s hard rock, crunchy Marshall guitars, heavy drums, soaring raspy male vocals." \
  --lyrics "[intro]\n(Guitar riff)\n\n[verse]\nWalking down the highway tonight...\n\n[chorus]\nRocking through the storm!\n[end]" \
  --duration 215.0 \
  --steps 20 \
  --output "./outputs/my_rock_song.wav"
```

### 4. Run an Automated Batch with Thermal Protection

When rendering multiple tracks, the GPU can accumulate heat. `batch_generator.py` executes each song in an isolated subprocess (ensuring 100% memory reclamation by the OS) and triggers a 5-minute cooldown pause between tracks:

```bash
python batch_generator.py \
  --tracks-file "./examples/sample_tracks.json" \
  --output-dir "./outputs" \
  --cooldown-sec 300
```

---

## 💡 Prompt Engineering & Structuring Tips

To get professional audio results from MiniMax-Music 3.0:

1. **Always declare musical attributes first:**
   `Basic Attributes: bpm is 128. key is E, and scale is minor.`
2. **Describe vocal timbre with anatomical precision:**
   `Vocal Gender & Timbre: Male lead singer with a gritty raspy baritone verse and high soaring chest screaming chorus.`
3. **Use structural arrangement markers in lyrics:**
   Use tags like `[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`, `[guitar solo]`, `[outro]`, `[fade out]`, `[end]`.
4. **Instrumental Cues in Parentheses:**
   Lines like `(Heavy down-picked thrash riff with double-bass kick)` help the language model position instruments and dynamic breakdowns.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
Totally free and permissive for personal, commercial, and research use.
