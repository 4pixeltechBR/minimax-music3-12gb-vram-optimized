<div align="center">

# 🎵 MiniMax-Music 3.0: 12GB VRAM Optimization Engine
### Motor de Otimização para GPUs de 12GB VRAM (RTX 3060 / 4070)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![VRAM: 12GB Tested](https://img.shields.io/badge/VRAM-12GB%20Target-success.svg)](#-benchmarks-8-faixas-produzidas)
[![Audio Demos](https://img.shields.io/badge/Audio%20Demos-8%20Tracks%20(320k)-brightgreen.svg)](#-demonstrações-em-áudio-mp3-320-kbps)

### 🌐 Selecione o Idioma / Select Language:
**[ 🇧🇷 Português (PT-BR) ](#-versão-em-português)** &nbsp;&nbsp;|&nbsp;&nbsp; **[ 🇺🇸 English (EN) ](#-english-version)**

</div>

---

## 🇧🇷 Versão em Português

[![License: MIT](https://img.shields.io/badge/Licença-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![VRAM: 12GB Testado](https://img.shields.io/badge/VRAM-12GB%20Validado-success.svg)](#benchmarks)
[![English Docs](https://img.shields.io/badge/Docs-English-blue.svg)](README.md)

> **Otimização de nível de produção para o modelo MiniMax-Music 3.0 em GPUs de 12GB de VRAM (NVIDIA RTX 3060, 4070).**  
> Reduz o tempo de geração de músicas completas (~3m35s de áudio) de **8.5 horas para apenas ~24 minutos** — uma aceleração de **mais de 19x (~95% de redução de espera)** sem nenhum swap para RAM via PCIe e com fidelidade vocal impecável.

---

## ⚡ O Desafio: Por que os Pipelines Padrão Falham?

O MiniMax-Music 3.0 é um pipeline avançado de duas etapas que combina um **Modelo de Linguagem de 8 bilhões de parâmetros (Qwen-8B)** para geração de tokens semânticos de áudio com um **Diffusion Transformer (DiT) de 2.4 bilhões de parâmetros** e um Vocoder acústico.

Executar a implementação original em placas com 12GB de VRAM causa gargalos severos:
1. **Gargalo de Troca de Camadas (Offload em RAM/Disco):** As implementações padrão ficam constantemente trocando tensores entre a VRAM e a RAM do sistema através do barramento PCIe, gerando áudio a lentos **~0.18 tokens/segundo** (~8 horas e 25 minutos por música).
2. **Degradação por Quantização INT8 no KV-Cache:** Tentar quantizar o KV-Cache para INT8 causa forte perda de frequências e perda de dinâmica vocal durante os primeiros 20 segundos ("ataque vocal anestesiado"), além de estourar a memória compartilhada no Windows.

### 🛠️ Nossa Solução Arquitetural

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASE 1: GERAÇÃO SEMÂNTICA (VRAM Pico: ~8.6 GB / Reservada: ~9.8 GB)         │
│                                                                             │
│  [Prompt + Letra] ──> [Qwen-8B em 4-bit NF4] ──> [KV Cache Nativo em BF16]  │
│                           (~4.4 GB na GDDR6)       (Ataque nítido e puro)   │
│                                      │                                      │
│                      Gera ~5.375 Tokens (4.87 tok/s)                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                      [TRANSIÇÃO TWO-PHASE NA VRAM]
                    (Descarrega Qwen & RVQ -> Libera ~8.6 GB)
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ FASE 2: DIFUSÃO ACÚSTICA & MASTER (VRAM Pico: ~5.0 GB)                      │
│                                                                             │
│  [Tokens Semânticos] ──> [DiT 2.4B Direto em CUDA] ──> [Vocoder em CUDA]    │
│                                 (6.5 minutos)               (14 segundos)   │
│                                      │                                      │
│               [Engine de Fade-Out Cosseno de Estúdio (5s)]                  │
│                                      │                                      │
│               Resultado: Áudio Master (44.1 kHz Estéreo WAV)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Quantização NF4 de 4-bits em Memória (BitsAndBytes):** Converte recursivamente as 253 camadas lineares do Qwen-8B para `BitsAndBytesLinear4bit` (NormalFloat4), encolhendo os pesos de 8.76 GB para ~4.40 GB na VRAM.
2. **KV-Cache Nativo em BF16:** Preserva as chaves e valores da atenção em `bfloat16` puro sem compressão, eliminando ruídos iniciais e garantindo dicção e potência vocal cristalinas desde o primeiro verso.
3. **Gerenciamento Two-Phase de VRAM:** Mantém o Modelo de Linguagem e o RVQ em GDDR6 rápida e os descarrega totalmente antes de subir o DiT de 2.4B direto para CUDA, sem fatiamento por PCIe.
4. **Engine Inteligente de Fade-Out em Cosseno:** Detecta a dinâmica do final da faixa e aplica atenuação suave nos últimos 5 segundos se necessário, eliminando cortes secos.

---

## 📊 Benchmarks: 8 Faixas Produzidas

Todas as faixas são produções completas de **até 3 minutos e 35 segundos (215 segundos)** criadas em uma única **NVIDIA GeForce RTX 3060 (12GB GDDR6)**:

| # | Título da Faixa | Gênero Musical | Duração do Áudio | Tempo de GPU | Taxa Semântica | Pico VRAM | Arquivo Master |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | **Shattered Glass Inside** | Nu-Metal / Melodic Alt-Rock | 3m 35s | 25.69 min | 4.80 tok/s | 8.99 GB | 36.21 MB WAV |
| **2** | **Shadows On The Dancefloor** | 80s Funk-Pop / Dance-Rock | 3m 28s | 24.65 min | 4.87 tok/s | 8.99 GB | 35.03 MB WAV |
| **3** | **Wharf Street Lanterns** | Roots Rock / British Blues | 3m 34s | 25.52 min | 4.86 tok/s | 8.94 GB | 35.97 MB WAV |
| **4** | **Crown of Thorns & Snow** | Symphonic Gothic Rock | 3m 35s | 24.37 min | 5.10 tok/s | 8.58 GB | 36.21 MB WAV |
| **5** | **Iron In The Blood** | 80s Thrash Metal / Heavy Metal | 3m 35s | 25.06 min | 4.94 tok/s | 8.58 GB | 36.21 MB WAV |
| **6** | **Antes Que O Dia Amanheça** | Pop Rock / Post-Punk Revival | 3m 35s | 25.90 min | 4.74 tok/s | 8.62 GB | 36.21 MB WAV |
| **7** | **One Breath Left** | Cinematic Hip-Hop / Rock-Rap | 2m 55s | 20.32 min | 4.95 tok/s | 8.63 GB | 29.36 MB WAV |
| **8** | **Circus of the Broken Clocks** | Avant-Garde Metal / Alt Nu-Metal | 3m 35s | 26.02 min | 4.72 tok/s | 8.62 GB | 36.21 MB WAV |

*A telemetria detalhada e medições acústicas estão documentadas em [BENCHMARKS.md](BENCHMARKS.md).*

---

## 🎧 Demonstrações em Áudio (MP3 320 kbps)

Ouça diretamente no navegador as 8 músicas completas geradas pelo pipeline em uma **NVIDIA RTX 3060 (12GB VRAM)**:

| # | Título da Faixa | Gênero / Identidade Sonora | Duração | Player no GitHub |
| :-: | :--- | :--- | :-: | :--- |
| **1** | **Shattered Glass Inside** | Nu-Metal / Melodic Alt-Rock | 3m 35s | [▶️ Ouvir Faixa](demos/shattered_glass_inside_master.mp3) |
| **2** | **Shadows On The Dancefloor** | 80s Funk-Pop / Dance-Rock | 3m 28s | [▶️ Ouvir Faixa](demos/shadows_on_the_dancefloor_master.mp3) |
| **3** | **Wharf Street Lanterns** | Roots Rock / British Blues | 3m 34s | [▶️ Ouvir Faixa](demos/wharf_street_lanterns_master.mp3) |
| **4** | **Crown of Thorns & Snow** | Symphonic Gothic Rock | 3m 35s | [▶️ Ouvir Faixa](demos/crown_of_thorns_and_snow_master.mp3) |
| **5** | **Iron In The Blood** | 80s Thrash Metal / Heavy Metal | 3m 35s | [▶️ Ouvir Faixa](demos/iron_in_the_blood_master.mp3) |
| **6** | **Antes Que O Dia Amanheça** | Pop Rock / Post-Punk Revival | 3m 35s | [▶️ Ouvir Faixa](demos/antes_que_o_dia_amanheca_master.mp3) |
| **7** | **One Breath Left** | Cinematic Hip-Hop / Rock-Rap | 2m 55s | [▶️ Ouvir Faixa](demos/one_breath_left_master.mp3) |
| **8** | **Circus of the Broken Clocks** | Avant-Garde Metal / Alt Nu-Metal | 3m 35s | [▶️ Ouvir Faixa](demos/circus_of_the_broken_clocks_master.mp3) |

> 💡 *Dica: Clique em "▶️ Ouvir Faixa" para abrir o player nativo de áudio do GitHub diretamente no seu navegador.*


---

## 🚀 Guia Rápido de Uso

### 1. Requisitos de Sistema
* **Sistema Operacional:** Windows 10/11 ou Linux
* **Python:** 3.10 ou 3.11
* **GPU:** Placa NVIDIA com pelo menos 12GB de VRAM (RTX 3060, 4070, A4000, 3080 12GB)
* **Memória RAM:** 32GB recomendados
* **CUDA / PyTorch:** CUDA 12.1+ / PyTorch 2.1+

### 2. Instalação

```bash
# Clone o repositório
git clone https://github.com/4pixeltechBR/minimax-music3-12gb-vram-optimized.git
cd minimax-music3-12gb-vram-optimized

# Crie e ative o ambiente virtual
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Geração de Música Individual

```bash
python generate_music.py \
  --prompt "Basic Attributes: bpm is 120. key is A, and scale is minor. 80s hard rock, crunchy Marshall guitars, heavy drums, soaring raspy male vocals." \
  --lyrics "[intro]\n(Guitar riff)\n\n[verse]\nWalking down the highway tonight...\n\n[chorus]\nRocking through the storm!\n[end]" \
  --duration 215.0 \
  --steps 20 \
  --output "./outputs/minha_musica.wav"
```

### 4. Orquestração em Lote com Proteção Térmica

Ao renderizar várias músicas em sequência, a GPU acumula calor. O script `batch_generator.py` roda cada música em um subprocesso isolado (garantindo que o sistema operacional recicle 100% da memória entre músicas) e aguarda 5 minutos de resfriamento entre as faixas:

```bash
python batch_generator.py \
  --tracks-file "./examples/sample_tracks.json" \
  --output-dir "./outputs" \
  --cooldown-sec 300
```

---

## 💡 Dicas de Engenharia de Prompt e Letra

Para obter a melhor fidelidade sonora no MiniMax-Music 3.0:

1. **Declare os parâmetros musicais no início:**
   `Basic Attributes: bpm is 128. key is E, and scale is minor.`
2. **Descreva a textura vocal com detalhes anatômicos:**
   `Vocal Gender & Timbre: Male lead singer with a gritty raspy baritone verse and high soaring chest screaming chorus.`
3. **Use marcadores estruturais nas letras:**
   Inclua tags como `[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`, `[guitar solo]`, `[outro]`, `[fade out]`, `[end]`.
4. **Cues instrumentais entre parênteses:**
   Inserções como `(Heavy down-picked thrash riff with double-bass kick)` direcionam os drops e dinâmica dos instrumentos.

---

## 📄 Licença

Distribuído sob a **Licença MIT**. Veja [`LICENSE`](LICENSE) para mais detalhes.
Totalmente livre e permissiva para uso comercial, pessoal e acadêmico.

---

## 🇺🇸 English Version


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

## 🎧 Audio Showcase (320 kbps MP3 Demos)

Listen to all 8 full-length master tracks directly in your browser, rendered autonomously on a consumer **NVIDIA RTX 3060 (12GB VRAM)**:

| # | Track Title | Sonic Identity / Genre | Audio Length | In-Browser Player |
| :-: | :--- | :--- | :-: | :--- |
| **1** | **Shattered Glass Inside** | Nu-Metal / Melodic Alt-Rock | 3m 35s | [▶️ Listen to Track](demos/shattered_glass_inside_master.mp3) |
| **2** | **Shadows On The Dancefloor** | 80s Funk-Pop / Dance-Rock | 3m 28s | [▶️ Listen to Track](demos/shadows_on_the_dancefloor_master.mp3) |
| **3** | **Wharf Street Lanterns** | Roots Rock / British Blues | 3m 34s | [▶️ Listen to Track](demos/wharf_street_lanterns_master.mp3) |
| **4** | **Crown of Thorns & Snow** | Symphonic Gothic Rock | 3m 35s | [▶️ Listen to Track](demos/crown_of_thorns_and_snow_master.mp3) |
| **5** | **Iron In The Blood** | 80s Thrash Metal / Heavy Metal | 3m 35s | [▶️ Listen to Track](demos/iron_in_the_blood_master.mp3) |
| **6** | **Antes Que O Dia Amanheça** | Pop Rock / Post-Punk Revival | 3m 35s | [▶️ Listen to Track](demos/antes_que_o_dia_amanheca_master.mp3) |
| **7** | **One Breath Left** | Cinematic Hip-Hop / Rock-Rap | 2m 55s | [▶️ Listen to Track](demos/one_breath_left_master.mp3) |
| **8** | **Circus of the Broken Clocks** | Avant-Garde Metal / Alt Nu-Metal | 3m 35s | [▶️ Listen to Track](demos/circus_of_the_broken_clocks_master.mp3) |

> 💡 *Tip: Click on "▶️ Listen to Track" to open GitHub's native HTML5 media player directly in your browser.*

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

