import os
import shutil
from pathlib import Path

def collect_thermal_mp4s(data_root, output_dir):
    """
    Copies all thermal_*/.mp4 files across session folders to a new directory
    and renames them as: sessionName_thermalCamID.mp4
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop over session folders
    for session_folder in data_root.glob("*session_*"):
        session_name = session_folder.name

        # Look for thermal_x folders
        for thermal_dir in session_folder.glob("thermal_*"):
            cam_name = thermal_dir.name
            mp4_files = list(thermal_dir.glob("*.mp4"))

            for mp4 in mp4_files:
                # Construct new filename
                new_name = f"{session_name}_{cam_name}.mp4"
                dest_path = output_dir / new_name

                print(f"📥 Copying {mp4.name} → {dest_path.name}")
                shutil.copy2(mp4, dest_path)

    print(f"\n✅ All thermal videos collected in: {output_dir}")

# Example usage
# collect_thermal_mp4s("data", "collected_thermal_videos")
