import subprocess
import os
import sys
import csv

def get_creation_time(filepath):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format_tags=creation_time',
             '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def extract_timestamps_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Provided path is not a folder: {folder_path}")
        return

    video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    video_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in video_extensions]

    print(f"\n📂 Scanning folder: {folder_path}")
    output = []
    for file in sorted(video_files):
        full_path = os.path.join(folder_path, file)
        print(f"📼 {file} ...", end=" ")
        timestamp = get_creation_time(full_path)
        print(f"{timestamp}")
        output.append((file, timestamp))

    # Save to CSV
    csv_path = os.path.join(folder_path, "video_timestamps.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "CreationTime"])
        writer.writerows(output)

    print(f"\n✅ Timestamps saved to {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python video_timestamps.py /path/to/video_folder")
    else:
        extract_timestamps_from_folder(sys.argv[1])
