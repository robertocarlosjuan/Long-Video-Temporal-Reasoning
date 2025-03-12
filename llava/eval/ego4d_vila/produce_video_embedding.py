# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is adopted from https://github.com/EvolvingLMMs-Lab/LongVA


import os
import argparse
import math

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

class VideoProcessor:
    def __init__(self, model_path, video_dir=None, batch_size=32, max_frame_num=7200, pooling_size=0, add_newline_token=False, output_dir=None):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(self.model_path)
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.max_frame_num = max_frame_num
        self.pooling_size = pooling_size
        self.add_newline_token = add_newline_token
        self.output_dir = output_dir
        self.loaded_model = False

    def load_video_model(self):
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path, self.model_name, None)
        model.config.image_aspect_ratio = "pad"
        model.config.mm_patch_merge_type = "flat"
        return tokenizer, model, image_processor, context_len

    def load_video_batches(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = round(vr.get_avg_fps())
        total_frame_num = min(len(vr), self.max_frame_num*fps)
        # Extracting at 1 fps
        frame_idx = [i for i in range(0, total_frame_num, fps)]
        print(f"Extracting {len(frame_idx)} frames at 1 fps...")
        for start_idx in range(0, len(frame_idx), self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_frame_num)
            frame_indices = frame_idx[start_idx:end_idx]
            batch_frames = vr.get_batch(frame_indices).asnumpy()
            yield batch_frames

    def get_video_features(self, video_path):
        if not self.loaded_model:
            self.tokenizer, self.model, self.image_processor, self.context_len = self.load_video_model()
            self.loaded_model = True
        # Process video in batches
        if self.video_dir is not None:
            video_path = os.path.join(self.video_dir, video_path)
        total_batches = (self.max_frame_num + self.batch_size - 1) // self.batch_size
        image_feature_list = []
        if self.add_newline_token:
            newline_token_embedding = self.model.model.image_newline
        with torch.inference_mode():
            for i, video_batch in tqdm(
                enumerate(self.load_video_batches(video_path)),
                total=total_batches,
                desc="Processing Video Batches",
            ):
                images = [Image.fromarray(frame).convert("RGB") for frame in video_batch]
                processed_images = process_images(images, self.image_processor, self.model.config).half()
                image_features = self.model.encode_images(processed_images, block_sizes=None)
                if self.pooling_size != 0:
                    B, _, F = image_features.shape

                    image_features_spatial = image_features.view(B, int(math.sqrt(_)), int(math.sqrt(_)), F).permute(
                        0, 3, 1, 2
                    )  # B, F, 24, 24
                    image_features_spatial_pool = torch.nn.functional.avg_pool2d(
                        image_features_spatial, self.pooling_size, self.pooling_size
                    )  # B, F, 12, 12
                    image_features = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()  # B, 144, F
                if self.add_newline_token:
                    image_features = torch.cat(
                        [
                            image_features,
                            newline_token_embedding.unsqueeze(0).expand(image_features.shape[0], 1, -1),
                        ],
                        dim=1,
                    )
                image_feature_list.append(image_features.to(torch.bfloat16).to("cpu"))
                if i > total_batches:
                    break
        return torch.cat(image_feature_list, dim=0)

    def precompute_video_features(self, video_uids):
        assert self.output_dir
        for video_uid in tqdm(video_uids):
            save_path = f"{self.output_dir}/{video_uid}.pt"
            if os.path.isfile(save_path):
                continue
            video_features = self.get_video_features(f"{video_uid}.mp4")
            video_features = video_features.to(torch.bfloat16)
            torch.save(video_features, save_path)
        return self.output_dir



def main(args):
    video_path = args.video_path
    model_path = args.model
    batch_size = 32
    max_frame_num = args.max_frame_num
    pooling_size = args.pooling_size
    add_newline_token = args.add_newline_token
    video_processor = VideoProcessor(model_path, batch_size, max_frame_num, pooling_size, add_newline_token)
    image_feature_list = video_processor.get_video_features(video_path)
    torch.save(image_feature_list, f"{args.output_dir}/video_embeddings.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Efficient-Large-Model/qwen2-7b-longvila-1M")
    parser.add_argument("--video_path", type=str, default="/nethome/che321/flash/datasets/Ego4D/v2/nlq_videos/full_scale/72295d26-19f7-4c6a-874e-85ba8654861e.mp4")
    parser.add_argument("--max_frame_num", type=int, default=7200)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nethome/che321/flash/VILA/llava/eval/ego4d_vila/data/video_embeddings/LongVILA-7B-1M",
    )
    parser.add_argument("--pooling_size", type=int, default=0)
    parser.add_argument("--add_newline_token", action="store_true")
    args = parser.parse_args()
    main(args)


