"""
MiniMax-Music 3.0: 12GB VRAM Optimized Production Generator
===========================================================
High-performance music synthesis pipeline for 12GB GPUs (RTX 3060, 4070).
Features:
  - 4-bit NF4 Dynamic Quantization for Qwen-8B backbone via BitsAndBytes.
  - Native BF16 KV Cache (preserves crystal-clear vocal dynamics, zero anesthesia).
  - Two-Phase VRAM Swapping (loads Language Model, unloads, then runs DiT 2.4B in Direct CUDA).
  - Smart Cosine Fade-Out Engine (guarantees smooth organic finishes without abrupt cuts).
"""

import os
import sys
import time
import json
import gc
import types
import argparse
import psutil
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from diffsynth.core.attention import attention as attn_mod
attn_mod.ATTENTION_IMPLEMENTATION = "torch"
from diffsynth.pipelines.minimax_music3 import MiniMaxMusic3Pipeline, ModelConfig
from tqdm import tqdm


# ==============================================================================
# 1. QUANTIZATION UTILITIES (4-BIT NF4 IN-MEMORY)
# ==============================================================================

def convert_linears_to_nf4(module: nn.Module) -> int:
    """
    Recursively replaces all nn.Linear layers with BitsAndBytesLinear4bit (NF4).
    Reduces memory footprint of Qwen-8B from ~8.8 GB down to ~4.4 GB in VRAM.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            w = child.weight.dequantize() if hasattr(child.weight, 'dequantize') else child.weight.data
            w = w.contiguous().to(device='cuda', dtype=torch.bfloat16)
            bias = child.bias.data.to(device='cuda', dtype=torch.bfloat16) if child.bias is not None else None

            new_linear = bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4'
            ).to('cuda')
            new_linear.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type='nf4').to('cuda')
            if bias is not None:
                new_linear.bias = nn.Parameter(bias, requires_grad=False)
            setattr(module, name, new_linear)
            count += 1
        else:
            count += convert_linears_to_nf4(child)
    return count


# ==============================================================================
# 2. SEMANTIC SYNTHESIS WITH NATIVE BF16 KV CACHE
# ==============================================================================

def patched_semantic_process_bf16_cache(self, pipe: MiniMaxMusic3Pipeline, text_ids, max_audio_duration, generator, progress_bar_cmd):
    """
    Patched autoregressive token generator that keeps the KV-Cache in unquantized BF16.
    Avoids quantization noise in early vocal tokens, preventing slurred/muffled vocal attacks.
    """
    if max_audio_duration <= 0:
        raise ValueError(f"`max_audio_duration` must be positive, got {max_audio_duration}")
    max_frames = min(int(max_audio_duration * self.frame_rate), self.max_audio_frames)
    backbone = pipe.text_encoder.model.model
    lm_head = pipe.text_encoder.model.lm_head
    vocab_size = pipe.text_encoder.config.vocab_size

    output = backbone(inputs_embeds=backbone.embed_tokens(text_ids), use_cache=True)
    past_key_values = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(vocab_size, dtype=torch.bool, device=text_ids.device)
    vocab_mask[self.audio_code_offset : self.audio_code_offset + self.semantic_vocab_size] = False
    vocab_mask[self.audio_end_token_id] = False

    frame_hiddens = []
    t_sem_start = time.time()
    natural_end = False

    print(f"\n[SEMANTIC] Generating up to {max_frames} frames ({max_audio_duration}s) with NF4 weights + Native BF16 KV Cache...", flush=True)

    for frame_index in progress_bar_cmd(
        range(max_frames + 1),
        desc="Generating semantic tokens",
        unit="tok",
        bar_format="{desc}: {n_fmt} tokens ({rate_fmt})",
    ):
        logits = lm_head(last_hidden).float()
        logits = logits.masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * self.ar_cfg_scale
        threshold = torch.topk(conditional, self.ar_cfg_top_k, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        sampled = self.sample_top_k(guided, generator)

        if int(sampled.item()) == self.audio_end_token_id:
            natural_end = True
            print(f"\n🎵 [Natural Ending] Model emitted audio_end_token at frame {frame_index} ({frame_index/self.frame_rate:.1f}s)!", flush=True)
            break

        semantic_code = sampled - self.audio_code_offset
        frame_codes, depth_hidden = self.generate_depth_codes(pipe, last_hidden, semantic_code.repeat(2), generator)

        if frame_index > 0:
            frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))

            if frame_index % 50 == 0 or len(frame_hiddens) >= max_frames:
                elapsed = time.time() - t_sem_start
                rate = frame_index / elapsed if elapsed > 0 else 0
                eta_s = (max_frames - frame_index) / rate if rate > 0 else 0
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(f"   Progress: {frame_index}/{max_frames} ({frame_index/max_frames*100:.1f}%) | {rate:.2f} tok/s | VRAM: {vram_gb:.2f} GB (Res: {vram_res:.2f} GB) | ETA: {eta_s/60:.1f} min", flush=True)

            if len(frame_hiddens) >= max_frames:
                break

        feedback = self.embed_audio_frame(pipe, frame_codes)
        output = backbone(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

    if not frame_hiddens:
        raise ValueError("MiniMax Music 3 generated 0 audio frames!")

    print(f"[OK] Semantic synthesis completed: {len(frame_hiddens)} frames generated (Natural end: {natural_end})", flush=True)
    return {"frame_hiddens": torch.stack(frame_hiddens, dim=1)}


# ==============================================================================
# 3. SMART STUDIO COSINE FADE-OUT
# ==============================================================================

def export_wav_with_smart_fadeout(audio, sample_rate, path, fade_out_sec=5.0):
    """
    Exports audio safely ensuring studio master normalization and natural cosine fade-out.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    if audio.ndim == 3:
        audio = audio.squeeze(0)
    if audio.ndim == 2 and audio.shape[0] in [1, 2] and audio.shape[1] > audio.shape[0]:
        audio = audio.T

    total_samples = len(audio)
    duration_s = total_samples / sample_rate

    tail_samples = int(min(2.0, duration_s) * sample_rate)
    tail = audio[-tail_samples:]
    tail_rms = float(np.sqrt(np.mean(tail**2)))

    fade_len = int(min(fade_out_sec, duration_s * 0.25) * sample_rate)
    if fade_len > 0:
        print(f"[AUDIO MASTER] Applying {fade_len/sample_rate:.1f}s studio cosine fade-out...", flush=True)
        t = np.linspace(0.0, np.pi / 2.0, fade_len)
        fade_curve = np.cos(t) ** 2
        if audio.ndim == 2:
            fade_curve = fade_curve[:, np.newaxis]
        audio[-fade_len:] = audio[-fade_len:] * fade_curve

    max_val = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
    if max_val > 1.0:
        audio = audio / max_val * 0.99

    sf.write(path, audio, sample_rate)
    print(f"[AUDIO MASTER] Master audio written: {path}", flush=True)


# ==============================================================================
# 4. MAIN GENERATION PIPELINE
# ==============================================================================

def generate_track(prompt: str, lyrics: str, duration_sec: float, output_path: str, seed: int = 42, steps: int = 20, cache_dir: str = "./cache"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    os.environ["MODELSCOPE_CACHE"] = os.path.join(cache_dir, "modelscope")
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")

    print("=" * 80)
    print(" 🎵 MINIMAX-MUSIC 3.0: TWO-PHASE 12GB OPTIMIZED INFERENCE")
    print(f" Target Duration: {duration_sec}s | Steps: {steps} | Seed: {seed}")
    print(f" Device: {torch.cuda.get_device_name(0)} ({round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)} GB VRAM)")
    print("=" * 80)

    t_start = time.time()

    lm_cuda_config = {
        "offload_dtype": None,
        "offload_device": None,
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    cpu_offload_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    print("\n[1/5] Loading MiniMax-Music 3.0 Pipeline...")
    pipe = MiniMaxMusic3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="language_model/model*.safetensors", **lm_cuda_config),
            ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="rvq_depth_decoder/diffusion_pytorch_model.safetensors", **lm_cuda_config),
            ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **cpu_offload_config),
            ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="condition_encoder/diffusion_pytorch_model.safetensors", **cpu_offload_config),
            ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="vocoder/diffusion_pytorch_model.safetensors", **cpu_offload_config),
        ],
        tokenizer_config=ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="tokenizer/"),
        vram_limit=None,
    )
    pipe.vram_management_enabled = False

    print("\n[2/5] Quantizing Qwen-8B Linear Layers to 4-bit NF4...")
    num_converted = convert_linears_to_nf4(pipe.text_encoder.model)
    gc.collect()
    torch.cuda.empty_cache()
    vram_post_quant = torch.cuda.memory_allocated() / (1024**3)
    print(f"[OK] {num_converted} layers converted to 4-bit NF4! Current VRAM: {vram_post_quant:.2f} GB")

    semantic_unit = pipe.units[1]
    semantic_unit.process = types.MethodType(patched_semantic_process_bf16_cache, semantic_unit)

    with torch.inference_mode():
        # Prompt Embeddings
        print("\n[3/5] Processing Text Prompt and Lyrics...")
        inputs_shared = {
            "prompt": prompt,
            "lyrics": lyrics,
            "max_audio_duration": float(duration_sec),
            "num_inference_steps": int(steps),
            "cfg_scale": 1.7,
            "generator": torch.Generator("cuda").manual_seed(seed),
            "rand_device": "cuda",
            "progress_bar_cmd": tqdm,
        }
        inputs_posi, inputs_nega = {}, {}
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(pipe.units[0], pipe, inputs_shared, inputs_posi, inputs_nega)

        # Semantic Synthesis
        t_sem = time.time()
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(pipe.units[1], pipe, inputs_shared, inputs_posi, inputs_nega)
        sem_dur = time.time() - t_sem
        num_tokens = inputs_shared["frame_hiddens"].shape[1]
        sem_rate = num_tokens / sem_dur if sem_dur > 0 else 0
        print(f"[OK] {num_tokens} tokens synthesized in {sem_dur:.1f}s ({sem_rate:.2f} tok/s)")

        # Two-Phase Transition
        print("\n>>> TWO-PHASE VRAM TRANSITION: Freeing Qwen-8B and RVQ from GPU...")
        del pipe.text_encoder
        del pipe.rvq_depth_decoder
        pipe.text_encoder = None
        pipe.rvq_depth_decoder = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f">>> VRAM cleanly recovered! Free: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")

        # Acoustic Diffusion in Direct CUDA
        print("\n[4/5] Running DiT 2.4B Acoustic Diffusion in Direct CUDA...")
        pipe.condition_encoder.to("cuda")
        pipe.dit.to("cuda")
        t_dit = time.time()
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(pipe.units[2], pipe, inputs_shared, inputs_posi, inputs_nega)
        dit_dur = time.time() - t_dit
        print(f"[OK] DiT diffusion completed in {dit_dur:.1f}s ({dit_dur/60:.2f} min)")

        # Vocoder
        pipe.dit.to("cpu")
        pipe.condition_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        print("\n[5/5] Synthesizing Audio via Vocoder...")
        pipe.vocoder.to("cuda")
        t_voc = time.time()
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(pipe.units[3], pipe, inputs_shared, inputs_posi, inputs_nega)
        voc_dur = time.time() - t_voc
        print(f"[OK] Vocoder completed in {voc_dur:.1f}s")

        # Export Master
        export_wav_with_smart_fadeout(inputs_shared["audio"], 44100, output_path, fade_out_sec=5.0)

    total_dur = time.time() - t_start
    audio_data, sr = sf.read(output_path)
    dur_actual = len(audio_data) / sr
    rms_overall = float(np.sqrt(np.mean(audio_data**2)))
    peak_val = float(np.max(np.abs(audio_data)))
    rms_db = 20 * np.log10(max(rms_overall, 1e-6))
    peak_db = 20 * np.log10(max(peak_val, 1e-6))

    print("\n" + "=" * 80)
    print(" 🎉 MASTER AUDIO GENERATION COMPLETED SUCCESSFULLY!")
    print(f" File: {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")
    print(f" Duration: {dur_actual:.1f}s | Total Generation Time: {total_dur/60:.2f} min")
    print(f" Semantic Speed: {sem_rate:.2f} tok/s | Acoustics: RMS {rms_db:.2f} dB, Peak {peak_db:.2f} dB")
    print("=" * 80)


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax-Music 3.0 Low-VRAM 12GB Production Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Detailed musical prompt (genre, instruments, vocal timbre, BPM, key)")
    parser.add_argument("--lyrics", type=str, default=" ", help="Song lyrics with structural tags ([intro], [verse], [chorus], etc.)")
    parser.add_argument("--duration", type=float, default=215.0, help="Target audio duration in seconds (default: 215.0)")
    parser.add_argument("--steps", type=int, default=20, help="DiT inference steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="./outputs/master.wav", help="Output WAV file path")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="HuggingFace/ModelScope model cache folder")

    args = parser.parse_args()
    generate_track(
        prompt=args.prompt,
        lyrics=args.lyrics,
        duration_sec=args.duration,
        output_path=args.output,
        seed=args.seed,
        steps=args.steps,
        cache_dir=args.cache_dir
    )
