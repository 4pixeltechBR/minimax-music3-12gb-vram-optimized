"""
MiniMax-Music 3.0: Batch Orchestrator with Thermal Protection
============================================================
Runs multiple track generations in isolated subprocesses with thermal cooldown
pauses between songs to prevent GPU thermal throttling.
"""

import os
import sys
import time
import json
import subprocess
import argparse


def get_gpu_temperature():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(out.strip())
    except Exception:
        return None


def run_cooldown(seconds=300):
    print("\n" + "#" * 80)
    print(f" ❄️ THERMAL COOLDOWN PAUSE: {seconds//60} MINUTES TO VENT GPU HEAT...")
    print("#" * 80)

    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        remaining = seconds - elapsed
        if remaining <= 0:
            break
        temp = get_gpu_temperature()
        temp_str = f"{temp}°C" if temp is not None else "N/A"
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        print(f"   [Cooling] Remaining: {mins:02d}m {secs:02d}s | Current GPU Temp: {temp_str}...", flush=True)
        time.sleep(min(30, remaining))

    final_temp = get_gpu_temperature()
    final_temp_str = f"{final_temp}°C" if final_temp is not None else "N/A"
    print(f"[OK] Cooldown completed! GPU stabilized at {final_temp_str}.\n", flush=True)


def run_batch(tracks_file: str, output_dir: str = "./outputs", cooldown_sec: int = 300, cache_dir: str = "./cache"):
    os.makedirs(output_dir, exist_ok=True)
    with open(tracks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", data) if isinstance(data, dict) else data
    total_tracks = len(tracks)
    python_bin = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generator_script = os.path.join(script_dir, "generate_music.py")

    print("=" * 80)
    print(f" 🚀 STARTING BATCH GENERATION: {total_tracks} TRACKS")
    print(f" Thermal cooldown between tracks: {cooldown_sec//60} minutes")
    print("=" * 80)

    completed = []
    t_batch_start = time.time()

    for i, track in enumerate(tracks):
        title = track.get("title", f"track_{i+1}")
        genre = track.get("genre", "Unknown")
        prompt = track.get("prompt", "")
        lyrics = track.get("lyrics", " ")
        duration = float(track.get("duration_sec", 215.0))
        steps = int(track.get("steps", 20))
        seed = int(track.get("seed", 42 + i))

        out_name = track.get("out_name", f"{title.lower().replace(' ', '_')}.wav")
        out_path = os.path.join(output_dir, out_name)

        print(f"\n>>> [{i+1}/{total_tracks}] Track: '{title}' ({genre})")

        # Resume/Idempotence check
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024 * 1024:
            print(f"⏩ [RESUME] Track already completed: '{out_path}'. Skipping to next...")
            completed.append({"title": title, "genre": genre, "file": out_name, "skipped": True})
            continue

        cmd = [
            python_bin, generator_script,
            "--prompt", prompt,
            "--lyrics", lyrics,
            "--duration", str(duration),
            "--steps", str(steps),
            "--seed", str(seed),
            "--output", out_path,
            "--cache-dir", cache_dir
        ]

        t_sub_start = time.time()
        ret = subprocess.run(cmd)
        sub_dur = time.time() - t_sub_start

        if ret.returncode != 0:
            print(f"❌ Error generating track {i+1} ('{title}'). Exit code: {ret.returncode}")
            break
        else:
            print(f"✅ Track {i+1}/{total_tracks} finished in {sub_dur/60:.2f} min!")
            completed.append({"title": title, "genre": genre, "file": out_name, "duration_sec": sub_dur})

        # Cooldown if not last track
        if i < total_tracks - 1 and cooldown_sec > 0:
            run_cooldown(cooldown_sec)

    total_time = time.time() - t_batch_start
    print("\n" + "=" * 80)
    print(f" 🎉 BATCH RUN FINISHED: {len(completed)}/{total_tracks} tracks processed in {total_time/60:.2f} min")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch music generator with thermal protection")
    parser.add_argument("--tracks-file", type=str, default="./examples/sample_tracks.json", help="Path to tracks JSON file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory for generated WAVs")
    parser.add_argument("--cooldown-sec", type=int, default=300, help="Cooling pause in seconds between tracks (default: 300)")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")

    args = parser.parse_args()
    run_batch(args.tracks_file, args.output_dir, args.cooldown_sec, args.cache_dir)
