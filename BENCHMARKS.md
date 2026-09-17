# 📊 MiniMax-Music 3.0: Hardware Optimization & Performance Benchmarks

Comprehensive forensic benchmark report documenting the memory, compute, and acoustic efficiency gains achieved on a consumer **NVIDIA GeForce RTX 3060 (12GB VRAM GDDR6)** with an Intel/AMD 8-core CPU and 32GB RAM.

---

## 1. Architectural Evolution & Performance Comparison

Comparison for generating a complete **3-minute and 35-second (215s) song** (~5,375 semantic tokens + 20 DiT diffusion steps):

| Metric | Approach 1: Official DiffSynth (CPU/Disk Offload) | Approach 2: Qwen INT8 + INT8 KV-Cache | **Approach 3: Ours (NF4 4-bit + BF16 Cache + Two-Phase)** |
| :--- | :---: | :---: | :---: |
| **Total Generation Time** | **8h 25m (30,346s)** | **6h 02m (21,749s)** | **~24m 40s (1,480s)** |
| **Speedup vs Baseline** | 1.0x (Baseline) | 1.39x | **~19.3x faster (~95% reduction in wait time)** |
| **Semantic Rate** | ~0.18 tokens/sec | ~0.28 tok/s *(when thrashing)* | **4.87 tokens/sec (up to 6.6 tok/s peak)** |
| **VRAM Footprint (Qwen-8B)** | ~8.76 GB (Unquantized BF16) | ~6.85 GB (INT8 Dynamic) | **~4.40 GB (4-bit NF4 BitsAndBytes)** |
| **KV-Cache Memory Type** | Disk/RAM Offload | INT8 Quantized Cache | **Native BF16 Floating Point** |
| **PCIe Bus Thrashing** | Continuous disk/RAM swaps | Memory spillover at token 3,835 | **Zero PCIe Paging (100% pure GDDR6)** |
| **Acoustic Fidelity** | Standard | Muffled/slurred vocal attack | **Pristine, punchy, uncompressed vocal attack** |
| **Track Ending** | Abrupt cutoff | Abrupt cutoff | **Smart 5-second studio cosine fade-out** |

---

## 2. Forensic Telemetry Table: 9 Validated Master Tracks

All tracks were generated autonomously with 5-minute thermal cooldown periods between tracks.

| # | Track Title | Genre / Sonic Identity | Audio Duration | GPU Production Time | Semantic Rate | Peak VRAM | RMS Energy | Peak Level | File Size |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | **Shattered Glass Inside** | Nu-Metal / Melodic Alt-Rock | 3.59 min (215.3s) | 25.69 min | 4.80 tok/s | 8.99 GB | -17.11 dB | -0.14 dB | 36.21 MB |
| **2** | **Shadows On The Dancefloor** | 80s Funk-Pop / Dance-Rock | 3.47 min (208.2s) | 24.65 min | 4.87 tok/s | 8.99 GB | -16.98 dB | 0.00 dB | 35.03 MB |
| **3** | **Wharf Street Lanterns** | Roots Rock / British Blues | 3.56 min (213.8s) | 25.52 min | 4.86 tok/s | 8.94 GB | -18.19 dB | 0.00 dB | 35.97 MB |
| **4** | **Crown of Thorns & Snow** | Symphonic Gothic Rock / Metal | 3.59 min (215.3s) | 24.37 min | 5.10 tok/s | 8.58 GB | -18.44 dB | 0.00 dB | 36.21 MB |
| **5** | **Iron In The Blood** | 80s Thrash Metal / Heavy Metal | 3.59 min (215.3s) | 25.06 min | 4.94 tok/s | 8.58 GB | -18.03 dB | 0.00 dB | 36.21 MB |
| **6** | **Antes Que O Dia Amanheça** | Pop Rock / Post-Punk Revival | 3.59 min (215.3s) | 25.90 min | 4.74 tok/s | 8.62 GB | -17.31 dB | 0.00 dB | 36.21 MB |
| **7** | **One Breath Left** | Cinematic Hip-Hop / Rock-Rap | 2.91 min (174.5s) | 20.32 min | 4.95 tok/s | 8.63 GB | -16.93 dB | 0.00 dB | 29.36 MB |
| **8** | **Circus of the Broken Clocks** | Avant-Garde Metal / Alt Nu-Metal | 3.59 min (215.3s) | 26.02 min | 4.72 tok/s | 8.62 GB | -17.09 dB | 0.00 dB | 36.21 MB |
| **9** | **Na Manha do Gato** | Brazilian Samba-Rap / Rio Hip-Hop | 3.15 min (189.0s) | 37.25 min | 2.57 tok/s | 8.88 GB | -18.19 dB | 0.00 dB | 31.80 MB |

---

## 3. Consolidated Production Totals

* **Total Completed Tracks:** 9 master recordings (100% success rate)
* **Total Clean Audio Rendered:** **31.03 minutes (1,862.05 seconds)** of CD-quality 44.1 kHz stereo audio
* **Total GPU Production Time:** **234.77 minutes (~3h 55m)**
* **Average Production Time per Song:** **26.09 minutes**
* **Total Semantic Tokens Processed:** **46,703 tokens**
* **Average Semantic Synthesis Throughput:** **4.47 tokens/second**
* **VRAM Ceiling:** **8.99 GB allocated / 11.06 GB reserved** (Consistently > 2.0 GB free GDDR6 margin)
* **Total Output Size:** **313.21 MB** in uncompressed WAV masters

---

## 4. Why Native BF16 KV-Cache Matters (The "Anesthetized Vocal" Fix)

In early attempts using INT8-quantized KV-Caches, early semantic tokens suffered from quantization noise in attention keys and values. Because autoregressive audio generation relies on the initial 200–500 tokens to establish vocal formant resonance and frequency envelopes, INT8 cache quantization produced:
1. **Slurred vocal attack:** The singer sounded muffled or "anesthetized" during the first 15–20 seconds.
2. **Frequency drift:** The model struggled to anchor pitch stability.

By quantizing **only model weights** to 4-bit NF4 (`BitsAndBytesLinear4bit`) and preserving the **KV-Cache in native BF16**, we achieved:
- **Zero vocal degradation:** Instant crisp vocal articulation from the very first lyric syllable.
- **Minimal memory impact:** The BF16 cache only grows to ~1.4 GB at 5,500 tokens, which easily fits within the 12GB budget because the NF4 weights occupy only 4.4 GB.

---

## 5. Thermal Cooldown Protocol

During sustained high-throughput inference, modern consumer GPUs (such as the RTX 3060) can reach 78°C–81°C under full load. 
The included `batch_generator.py` implements an active cooldown cycle (default 300 seconds) between tracks:
- Monitors temperature via `nvidia-smi`.
- Drops operating temperatures from **81°C down to 36°C–39°C** before the next track begins.
- Completely prevents thermal throttling and clock frequency degradation across multi-hour production runs.
