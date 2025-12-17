#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_ffmpeg_scale(src: Path, dst: Path, width: int, height: int, codec: str = "libx264", crf: int = 23,
                     preset: str = "veryfast") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale={width}:{height}",
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def list_head_camera_videos(root: Path, key_candidates: list[str]) -> tuple[str | None, list[Path]]:
    videos_dir = root / "videos"
    if not videos_dir.is_dir():
        return None, []

    found_key = None
    video_files: list[Path] = []
    for key in key_candidates:
        pattern = str(videos_dir / "chunk-*" / key / "episode_*.mp4")
        files = sorted(Path(p) for p in glob(pattern))
        if files:
            found_key = key
            video_files = files
            break
    return found_key, video_files


def update_info_json(info_path: Path, feature_key: str, height: int, width: int, codec_name: str | None = None) -> None:
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    if feature_key not in features:
        raise KeyError(f"Feature '{feature_key}' not found in {info_path}")

    # Update shape [H, W, C]
    features[feature_key]["shape"] = [height, width, 3]

    # Update video info block if present
    info_block = features[feature_key].get("info", {})
    info_block["video.height"] = height
    info_block["video.width"] = width
    if codec_name:
        info_block["video.codec"] = codec_name
    features[feature_key]["info"] = info_block

    backup = info_path.with_suffix(info_path.suffix + ".backup")
    if not backup.exists():
        shutil.copy2(info_path, backup)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Downscale head_camera videos and update info.json")
    parser.add_argument("--dataset_root", type=str, 
                        default="/DATA/disk0/haoran/placemouse_datasets",
                        help="Path to dataset root (e.g., /home/you/project/lerobot/h265_datasets)")
    parser.add_argument("--target_height", type=int, default=1080)
    parser.add_argument("--target_width", type=int, default=1920)
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite original files in-place")
    parser.add_argument("--workers", type=int, 
                        # default=min(8, os.cpu_count() or 4))
                        default=2)
    parser.add_argument("--codec", type=str, default="libx264", 
                        help="ffmpeg video codec (e.g., libx264, libaom-av1)")
    parser.add_argument("--crf", type=int, default=23, help="ffmpeg CRF value")
    parser.add_argument("--preset", type=str, default="veryfast", 
                        help="ffmpeg preset")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")

    key_candidates = [
        "observation.images.head_camera",
        "observation.images.head",
    ]
    feature_key, video_files = list_head_camera_videos(dataset_root, key_candidates)
    if not feature_key or not video_files:
        raise FileNotFoundError("No head camera videos found. Checked keys: " + ", ".join(key_candidates))

    tmp_suffix = ".resized.tmp.mp4"

    def process_one(mp4_path: Path) -> tuple[Path, bool, str | None]:
        tmp_out = mp4_path.with_suffix(mp4_path.suffix + tmp_suffix)
        run_ffmpeg_scale(
            src=mp4_path,
            dst=tmp_out,
            width=args.target_width,
            height=args.target_height,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
        )
        if args.overwrite:
            shutil.move(str(tmp_out), str(mp4_path))
            return mp4_path, True, None
        else:
            dst_out = mp4_path.with_name(mp4_path.stem + f"_{args.target_height}x{args.target_width}.mp4")
            shutil.move(str(tmp_out), str(dst_out))
            return dst_out, False, None

    # Downscale in parallel
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, p): p for p in video_files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                out_path, overwritten, _ = fut.result()
                action = "overwritten" if overwritten else "written"
                print(f"[{action}] {out_path}")
            except subprocess.CalledProcessError as e:
                msg = f"[ffmpeg-failed] {p}: {e.stderr.decode(errors='ignore')[:512]}"
                print(msg)
                errors.append(msg)
            except Exception as e:  # noqa: BLE001
                msg = f"[failed] {p}: {e}"
                print(msg)
                errors.append(msg)

    if errors:
        print(f"Completed with {len(errors)} errors. See logs above.")

    # Update info.json
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        codec_name = "h264" if args.codec == "libx264" else ("av1" if args.codec == "libaom-av1" else None)
        update_info_json(
            info_path=info_path,
            feature_key=feature_key,
            height=args.target_height,
            width=args.target_width,
            codec_name=codec_name,
        )
        print(f"Updated {info_path} for feature '{feature_key}' to shape [{args.target_height}, {args.target_width}, 3]")
    else:
        print(f"Warning: info.json not found at {info_path}, skipped metadata update.")


if __name__ == "__main__":
    main()

# 示例命令
# conda activate base
# python3 examples/teleavatar/downscale_head_camera_videos.py --overwrite

