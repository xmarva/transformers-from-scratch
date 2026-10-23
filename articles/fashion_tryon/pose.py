"""Find the body in two of the project's photos with BlazePose (MediaPipe's pose landmarker), a model made to run on
phones (Bazarevsky et al., 2020): the person of the try-on example, standing, and a person crouching (data/crouch.png,
the left half of the project's DensePose picture). For each it gives 33 keypoints in the photo and the same 33 points
in 3D, in metres around the hips. Writes data/keypoints.json, which the figures read; MediaPipe itself is not a
dependency of the course.

    curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
    uv run --no-project --python 3.12 --with mediapipe --with pillow python articles/fashion_tryon/pose.py \\
        --model pose_landmarker_heavy.task
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
PANEL = 342  # the person is the first of the example's three panels


def main():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="pose_landmarker_heavy.task")
    model = parser.parse_args().model
    photos = {
        "standing": np.asarray(Image.open(DATA / "try_on_example.png").convert("RGB"))[:, :PANEL],
        "crouching": np.asarray(Image.open(DATA / "crouch.png").convert("RGB")),
    }
    options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_poses=1)
    out = {}
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for name, photo in photos.items():
            photo = np.ascontiguousarray(photo)
            found = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=photo))
            height, width = photo.shape[:2]
            out[name] = {
                "image": [[p.x * width, p.y * height, p.visibility] for p in found.pose_landmarks[0]],
                "world": [[p.x, p.y, p.z] for p in found.pose_world_landmarks[0]],
                "size": [width, height],
            }
    (DATA / "keypoints.json").write_text(json.dumps(out, indent=1))
    print(f"keypoints of {', '.join(out)} written to {DATA / 'keypoints.json'}")


if __name__ == "__main__":
    main()
