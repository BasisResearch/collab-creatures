# rotate_crop_gui.py
import cv2
import argparse
import os

def rotate_image(img, angle):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to load video.")
        return

    # --- Step 1: Trackbar rotation ---
    def on_trackbar(val):
        angle = cv2.getTrackbarPos("Angle", "Rotate") - 180
        rotated = rotate_image(frame, angle)
        cv2.imshow("Rotate", rotated)

    cv2.namedWindow("Rotate")
    cv2.createTrackbar("Angle", "Rotate", 180, 360, on_trackbar)
    on_trackbar(180)

    print("🌀 Adjust rotation. Press any key when satisfied...")
    cv2.waitKey(0)
    angle = cv2.getTrackbarPos("Angle", "Rotate") - 180
    cv2.destroyWindow("Rotate")
    print(f"✅ Selected angle: {angle}°")

    # --- Step 2: Select ROI ---
    rotated = rotate_image(frame, angle)
    roi = cv2.selectROI("Crop", rotated, showCrosshair=True)
    cv2.destroyWindow("Crop")
    x, y, w, h = map(int, roi)
    print(f"✅ Selected crop area: x={x}, y={y}, w={w}, h={h}")

    # --- Step 3: Process video ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("💾 Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated = rotate_image(frame, angle)
        cropped = rotated[y:y+h, x:x+w]
        out.write(cropped)

    cap.release()
    out.release()
    print(f"🎉 Saved adjusted video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save output video")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
    else:
        main(args.input, args.output)
