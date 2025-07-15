import os
import re
import tempfile
import subprocess
import numpy as np
from libjpeg import decode
from math import exp, sqrt, log
import exiftool
from pathlib import Path

MAGIC_SEQ = re.compile(b"\x46\x46\x46\x00\x52\x54")
EXIFTOOL_PATH = "/opt/homebrew/bin/exiftool"  # Update this if needed

class CSQReader:
    """
    Reader for FLIR .csq thermal video files.
    Extracts thermal image frames using ExifTool and raw temperature conversion.
    """
    def __init__(self, filename, blocksize=1_000_000):
        self.reader = open(filename, "rb")
        self.blocksize = blocksize
        self.leftover = b""
        self.imgs = []
        self.index = 0
        self.nframes = None

        if not os.path.exists(EXIFTOOL_PATH):
            raise FileNotFoundError(f"ExifTool not found at {EXIFTOOL_PATH}")

        self.et = exiftool.ExifTool(executable=EXIFTOOL_PATH)
        self.etHelper = exiftool.ExifToolHelper(executable=EXIFTOOL_PATH)
        self.et.run()

    def _populate_list(self):
        self.imgs = []
        self.index = 0
        x = self.reader.read(self.blocksize)
        if not x:
            return
        matches = list(MAGIC_SEQ.finditer(x))
        if not matches:
            return
        start = matches[0].start()
        if self.leftover:
            self.imgs.append(self.leftover + x[:start])
        if len(matches) < 2:
            return
        for m1, m2 in zip(matches, matches[1:]):
            self.imgs.append(x[m1.start():m2.start()])
        self.leftover = x[matches[-1].start():]

    def next_frame(self):
        if self.index >= len(self.imgs):
            self._populate_list()
            if not self.imgs:
                return None
        im = self.imgs[self.index]
        raw, metadata = extract_data(im, self.etHelper)
        thermal_im = raw2temp(raw, metadata[0])
        self.index += 1
        return thermal_im

    def skip_frame(self):
        if self.index >= len(self.imgs):
            self._populate_list()
            if not self.imgs:
                return False
        self.index += 1
        return True

    def count_frames(self):
        self.nframes = 0
        while self.skip_frame():
            self.nframes += 1
        self.reset()
        return self.nframes

    def get_metadata(self):
        if self.index >= len(self.imgs):
            self._populate_list()
            if not self.imgs:
                return None
        im = self.imgs[self.index]
        _, metadata = extract_data(im, self.etHelper)
        return metadata

    def reset(self):
        self.reader.seek(0)
        self.leftover = b""
        self.imgs = []
        self.index = 0

    def close(self):
        self.reader.close()


def extract_data(bin_data, etHelper):
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bin_data)
        fp.flush()
        metadata = etHelper.get_metadata(fp.name)
        binary = subprocess.check_output([EXIFTOOL_PATH, "-b", "-RawThermalImage", fp.name])
        raw = decode(binary)
    return raw, metadata


def raw2temp(raw, metadata):
    E = metadata["FLIR:Emissivity"]
    OD = metadata["FLIR:ObjectDistance"]
    RTemp = metadata["FLIR:ReflectedApparentTemperature"]
    ATemp = metadata["FLIR:AtmosphericTemperature"]
    IRWTemp = metadata["FLIR:IRWindowTemperature"]
    IRT = metadata["FLIR:IRWindowTransmission"]
    RH = metadata["FLIR:RelativeHumidity"]
    PR1 = metadata["FLIR:PlanckR1"]
    PB = metadata["FLIR:PlanckB"]
    PF = metadata["FLIR:PlanckF"]
    PO = metadata["FLIR:PlanckO"]
    PR2 = metadata["FLIR:PlanckR2"]
    ATA1 = float(metadata["FLIR:AtmosphericTransAlpha1"])
    ATA2 = float(metadata["FLIR:AtmosphericTransAlpha2"])
    ATB1 = float(metadata["FLIR:AtmosphericTransBeta1"])
    ATB2 = float(metadata["FLIR:AtmosphericTransBeta2"])
    ATX = metadata["FLIR:AtmosphericTransX"]

    emiss_wind = 1 - IRT
    refl_wind = 0
    h2o = (RH / 100) * exp(
        1.5587 + 0.06939 * ATemp - 0.00027816 * ATemp**2 + 0.00000068455 * ATemp**3
    )
    tau = lambda d: ATX * exp(-sqrt(d) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
        -sqrt(d) * (ATA2 + ATB2 * sqrt(h2o))
    )
    tau1 = tau(OD / 2)
    tau2 = tau(OD / 2)

    def radiance(T):
        return PR1 / (PR2 * (exp(PB / (T + 273.15)) - PF)) - PO

    raw_obj = (
        raw / E / tau1 / IRT / tau2
        - (1 - tau1) / E / tau1 * radiance(ATemp)
        - (1 - tau2) / E / tau1 / IRT / tau2 * radiance(ATemp)
        - emiss_wind / E / tau1 / IRT * radiance(IRWTemp)
        - (1 - E) / E * radiance(RTemp)
        - refl_wind / E / tau1 / IRT * radiance(RTemp)
    )
    temp_C = PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15

    return temp_C

def process_frame(frame, vmin, vmax):
    frame = np.clip(frame, vmin, vmax)
    return np.uint8((frame - vmin) / (vmax - vmin) * 255)

def choose_vmin_vmax(path, default_vmin=-5, default_vmax=37):
    frame_collection = []

    path = Path(path)
    if path.is_file() and path.suffix == ".csq":
        reader = CSQReader(str(path))
        frame = reader.next_frame()
        if frame is not None:
            frame_collection.append(frame.flatten())
    else:
        for folder in os.listdir(path):
            if folder.startswith("FLIR"):
                full_path = os.path.join(path, folder)
                for f_name in os.listdir(full_path):
                    if f_name.endswith(".csq"):
                        reader = CSQReader(os.path.join(full_path, f_name))
                        frame = reader.next_frame()
                        if frame is not None:
                            frame_collection.append(frame.flatten())

    if not frame_collection:
        return default_vmin, default_vmax

    all_pixels = np.concatenate(frame_collection)
    vmin = max(default_vmin, np.round(np.percentile(all_pixels, 0.1)))
    vmax = min(default_vmax, np.round(np.percentile(all_pixels, 99.999)))
    return vmin, vmax

def estimate_duration(reader: CSQReader, fps: float = 30.0) -> float:
    """
    Estimates duration in seconds by counting frames and dividing by fps.
    """
    print("⏱ Counting frames for duration estimate...")
    total_frames = reader.count_frames()
    duration = total_frames / fps
    print(f"→ {total_frames} frames ≈ {duration:.2f} seconds at {fps:.1f} fps")
    reader.reset()
    return duration
