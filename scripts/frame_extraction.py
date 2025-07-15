import os
import cv2
from tqdm import tqdm
from pathlib import Path

def video_to_frames(video_path: Path, out_dir: Path, prefix="frame", target_fps=15):
    """Extracts frames from video at target_fps and saves to directory."""
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = round(native_fps / target_fps)

    idx, saved = 0, 0
    with tqdm(total=total, desc=f"Extracting {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                fname = f"{prefix}_{saved:06d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved += 1
            idx += 1
            pbar.update(1)

    cap.release()
    print(f"✅ Extracted {saved} frames at {target_fps} FPS to {out_dir}\n")


def extract_frames_from_folder(videos_dir: Path, output_root: Path, target_fps=15):
    """Iterates through all .mp4 files and extracts frames into per-video folders."""
    videos = sorted(videos_dir.glob("time_*.mp4"))
    if not videos:
        print(f"❌ No .mp4 files found in {videos_dir}")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    for video in videos:
        video_stem = video.stem
        out_dir = output_root / video_stem
        out_dir.mkdir(exist_ok=True)
        print(f"📽️ Processing {video.name}")
        video_to_frames(video, out_dir, target_fps)

# # 🧪 Example usage
# if __name__ == "__main__":
#     VIDEO_FOLDER = Path("path/to/video_folder")         # folder containing your .mp4 videos
#     OUTPUT_FOLDER = Path("extracted_frames")            # root directory for output frame folders
#     extract_frames_from_folder(VIDEO_FOLDER, OUTPUT_FOLDER)
