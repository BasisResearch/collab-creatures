from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cv2
import shutil
from PIL import Image
from io import BytesIO
from typing import Tuple
from Desktop.labeling_data.yolo.CSQ_reader import CSQReader, choose_vmin_vmax 


def render_frame_with_colorbar(frame: np.ndarray, vmin: float, vmax: float, color: str, figsize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Renders a thermal frame with a colorbar and returns it as a numpy RGB image.
    """
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize)
    plt.axis("off")
    im = ax.imshow(frame, cmap=color, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=12)
    fig.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img.convert("RGB"))
    plt.close(fig)
    return img_array


def export_thermal_video(reader: CSQReader, out_path: Path, vmin: float, vmax: float, color: str, max_frames: int = 30):
    """
    Converts a sequence of thermal frames into an mp4 video with a fixed colormap.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = reader.next_frame()
    img = render_frame_with_colorbar(frame, vmin, vmax, color)
    height, width, _ = img.shape
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for _ in range(max_frames - 1):
        frame = reader.next_frame()
        if frame is None:
            break
        img = render_frame_with_colorbar(frame, vmin, vmax, color)
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✅ Exported: {out_path}")


# def process_directory(folder_path: str, max_frames: int = 10):
#     """
#     Walks through a folder of .csq files, previews each, and exports videos as thermal_<index>.mp4.
#     """
#     folder = Path(folder_path)
#     csq_files = sorted(folder.glob("*.csq"))
#     print(f"✅ Found {len(csq_files)} .csq files in {folder}")

#     for idx, path in enumerate(csq_files):
#         print(f"\n[{idx}] Previewing {path.name}")
#         reader = CSQReader(str(path))
#         frame = reader.next_frame()

#         # Plot first frame and ask user for vmin/vmax
#         plt.imshow(frame, cmap="hot")
#         plt.title(f"File: {path.name}")
#         plt.colorbar(label="Temperature (°C)")
#         plt.show()

#         vmin = float(input(f"Enter vmin for {path.name}: "))
#         vmax = float(input(f"Enter vmax for {path.name}: "))

#         out_path = folder / f"thermal_{idx}.mp4"
#         reader.reset()
#         export_thermal_video(reader, out_path, vmin, vmax, max_frames=max_frames)
#         reader.close()


# def process_directory(folder_path, out_path, color='binary', preview=True, max_frames=300):
#     folder_path = Path(folder_path)
#     csq_files = sorted(folder_path.glob("*.csq"))
#     print(f"Found {len(csq_files)} .csq files in {folder_path}")

#     for idx, file_path in enumerate(csq_files):
#         print(f"\nProcessing [{idx}]: {file_path.name}")

#         reader = CSQReader(str(file_path))

#         if preview:
#             frame = reader.next_frame()
#             sns.set_style("ticks")
#             fig, ax = plt.subplots()
#             plt.axis("off")
#             im = ax.imshow(frame, cmap=color)
#             plt.colorbar(im, ax=ax)
#             plt.title(f"Preview for {file_path.name}")
#             plt.show()

#             vmin = float(input("Enter vmin: "))
#             vmax = float(input("Enter vmax: "))
#         else:
#             print("Auto-detecting vmin/vmax...")
#             vmin, vmax = choose_vmin_vmax(file_path)
#             print(f"→ Using vmin={vmin}, vmax={vmax}")

#         reader.reset()
#         out_path = Path(out_path)
#         out_path.mkdir(parents=True, exist_ok=True)  # ensure base dir exists
#         out_file = out_path / f"thermal_{idx}.mp4"   # this is the *file*, not dir
#         export_thermal_video(reader, out_file, vmin, vmax, color=color, max_frames=max_frames)


def process_directory(folder_path, out_path, color='binary', preview=True, max_frames=300):
    folder_path = Path(folder_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    csq_files = sorted(folder_path.glob("*.csq"))
    print(f"Found {len(csq_files)} .csq files in {folder_path}")

    for idx, file_path in enumerate(csq_files):
        print(f"\nProcessing [{idx}]: {file_path.name}")

        reader = CSQReader(str(file_path))

        if preview:
            frame = reader.next_frame()
            sns.set_style("ticks")
            fig, ax = plt.subplots()
            plt.axis("off")
            im = ax.imshow(frame, cmap=color)
            plt.colorbar(im, ax=ax)
            plt.title(f"Preview for {file_path.name}")
            plt.show()

            vmin = float(input("Enter vmin: "))
            vmax = float(input("Enter vmax: "))
        else:
            print("Auto-detecting vmin/vmax...")
            vmin, vmax = choose_vmin_vmax(file_path)
            print(f"→ Using vmin={vmin}, vmax={vmax}")

        # Ask user if they want to save this file
        save_choice = input(f"Do you want to save the video and copy the .csq file for {file_path.name}? (y/n): ").strip().lower()
        if save_choice != 'y':
            print(f"⏭️ Skipping save for {file_path.name}")
            reader.close()
            continue

        # Copy the .csq file
        copied_csq_path = out_path / file_path.name
        shutil.copy2(file_path, copied_csq_path)
        print(f"📁 Copied {file_path.name} to {copied_csq_path}")

        # Export the video
        reader.reset()
        out_file = out_path / f"thermal_{idx}.mp4"
        export_thermal_video(reader, out_file, vmin, vmax, color=color, max_frames=max_frames)
        reader.close()

# Example usage in a notebook:
# from thermal_video_converter import process_directory
# process_directory("videos/526")
