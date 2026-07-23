# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Small multimodal loading helpers used by the SFT dataset.

Heavy Qwen video/image utilities are imported lazily so importing the SFT
trainer does not initialize a vision stack or require real media files.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

from PIL import Image


def process_image(image: dict[str, Any] | str | Image.Image, image_patch_size: int = 14) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        image = {"image": image}
    if not isinstance(image, dict):
        raise TypeError(f"image must be a mapping, path, or PIL image; got {type(image).__name__}")

    payload = dict(image)
    if "bytes" in payload:
        if "image" in payload:
            raise ValueError("image payload cannot contain both bytes and image")
        payload["image"] = Image.open(BytesIO(payload.pop("bytes"))).convert("RGB")

    try:
        from qwen_vl_utils import fetch_image

        try:
            result = fetch_image(payload, image_patch_size=image_patch_size)
        except TypeError:
            result = fetch_image(payload)
        return result.convert("RGB") if isinstance(result, Image.Image) else result
    except ImportError:
        source = payload.get("image") or payload.get("image_url")
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, str) and not source.startswith(("http://", "https://", "data:")):
            if source.startswith("file://"):
                source = source[7:]
            return Image.open(source).convert("RGB")
        raise


VIDEO_FORMAT_HELP = """Video input must be a qwen-vl-utils mapping, for example
{"video": ["file:///path/to/frame1.jpg", "file:///path/to/frame2.jpg"]}.
"""


def process_video(
    video: dict[str, Any],
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
):
    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    if nframes is not None and fps is not None:
        raise ValueError("nframes and fps are mutually exclusive")
    payload = dict(video)
    if "nframes" not in payload and "fps" not in payload:
        if nframes is not None:
            payload["nframes"] = nframes
        elif fps is not None:
            payload["fps"] = fps
    if fps_min_frames is not None:
        payload["min_frames"] = fps_min_frames
    if fps_max_frames is not None:
        payload["max_frames"] = fps_max_frames

    from qwen_vl_utils import fetch_video

    return fetch_video(
        payload,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_video_sample_fps,
        return_video_metadata=return_video_metadata,
    )
