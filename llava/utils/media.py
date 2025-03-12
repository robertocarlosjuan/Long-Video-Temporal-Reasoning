import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import requests
from transformers import PretrainedConfig

from llava.constants import MEDIA_TOKENS
from llava.media import Image, Video
from llava.utils import make_list
from llava.utils.logging import logger

__all__ = ["extract_media"]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image


def _load_video(video: Video, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    video_path = video.path
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    max_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while max_frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, max_frame_count - 1)
        if vidcap.grab():
            break
        max_frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    start_timestamp = 0 if video.start_timestamp == -1 else video.start_timestamp
    if video.end_timestamp == -1:
        end_timestamp = max_frame_count/fps if fps else 0
    else:
        end_timestamp = video.end_timestamp
    frame_count_start = fps * start_timestamp
    frame_count_end =  fps * end_timestamp
    if frame_count_start > max_frame_count or frame_count_end > max_frame_count:
        raise ValueError(f"Requested {video.start_timestamp} to {video.end_timestamp} is impossible for video of duration {int(max_frame_count/fps)}")

    # Extract frames uniformly
    # 
    # Extract fps = 1
    if (frame_count_end-frame_count_start+1)//fps <= num_frames:
        indices = [i*fps for i in range(int(start_timestamp), int(end_timestamp)+1)]
        print(f"Extracting {len(indices)} frames at 1 fps")
    else:
        print(f"Extracting {num_frames} frames uniformly")
        indices = np.round(np.linspace(frame_count_start, frame_count_end-1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            logger.warning(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        logger.warning("Extracting frames from video with specified FPS is not supported yet. Ignored.")

    frames = _load_video(video, num_frames=num_frames)
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        logger.warning(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    media["video"].append(_extract_video(part, config))
                text += MEDIA_TOKENS["video"]
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media


