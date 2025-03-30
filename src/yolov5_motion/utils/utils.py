import cv2
from pathlib import Path
from decord import VideoReader
from decord import cpu, gpu
from matplotlib import pyplot as plt
import numpy as np


def parse_video_for_frames(video_path: Path, num_frames: int = np.inf, seconds_per_frame: int = 1) -> None:
    save_path = video_path.parent.parent.joinpath('images')

    save_path.mkdir(exist_ok=True)
    count_frames = 0

    vr = VideoReader(video_path.as_posix(), ctx=cpu(0), height=360, width=640)

    num_frames = min(num_frames, len(vr))

    try:
        batch = vr.get_batch(np.arange(0, len(vr), seconds_per_frame * vr.get_avg_fps())[:num_frames])
    except Exception as e:
        print(e)
        return
    for frame in batch.asnumpy():

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imwrite((save_path / f"{video_path.stem}{count_frames}.jpg").as_posix(), frame)

        count_frames += 1

        if count_frames >= num_frames:
            break
