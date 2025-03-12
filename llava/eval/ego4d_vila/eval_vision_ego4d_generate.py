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
# This file is modified from https://github.com/EvolvingLMMs-Lab/LongVA

import argparse
import gc
import copy
import glob
import json
import os
import random
import sys
from pathlib import Path
from collections import OrderedDict, defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from accelerate import Accelerator
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, GenerationConfig

from llava.eval.ego4d_vila.zigzag_ring_attn.modeling_qwen2 import Qwen2ForCausalLM_RingAttn
from llava.eval.ego4d_vila.zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_llama
from llava.eval.ego4d_vila.zigzag_ring_attn.prepare_inputs import prepare_zigzag_ring_attn_inputs
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.model import LlavaLlamaModel
from llava.model.utils import get_model_config
from llava.utils.tokenizer import tokenize_conversation
import torch.nn.functional as F
from llava.media import Video

from produce_video_embedding import VideoProcessor

apply_zigzag_ring_attn_monkey_patch_llama()

SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompt_templates = {
    "mistral": {"preprompt": "<s>[INST]", "postprompt": " [/INST]"},
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:",
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

def default_generation_config(tokenizer):
    generation_config = copy.deepcopy(GenerationConfig())
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token")
    if generation_config.max_length == GenerationConfig().max_length:
        generation_config.max_length = tokenizer.model_max_length
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if generation_config.bos_token_id is None:
        generation_config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    return generation_config

def undo_extract_local(gathered_value, world_size, dim=1):
    value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
    reordered_chunks = [None] * (2 * world_size)
    for i in range(world_size):
        reordered_chunks[i] = value_chunks[i * 2]
        reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
    return torch.cat(reordered_chunks, dim=dim)

def eval_forward(accelerator, model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer):
    # first append answer_embeds to input_embeds
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    # second pad input_embeds to the multiple of accelerator.num_processes
    pad_tensor = (
        torch.tensor(
            [pad_id] * ((accelerator.num_processes * 2) - input_embeds.shape[1] % (accelerator.num_processes * 2))
        )
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(-1, -1, input_embeds.shape[-1])
    )  # .to(accelerator.device)
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    )  # .to(accelerator.device)
    accelerator.print(input_embeds.shape)
    prepared = prepare_zigzag_ring_attn_inputs(
        input_embeds,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    local_input_embeds = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    with torch.inference_mode():
        logits = model(
            inputs_embeds=local_input_embeds,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        pred = logits.argmax(dim=-1)

    # gather all logits using accelerator.gather
    

    correct = False

    gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
    # undo extract local on the gathered logits
    pred = undo_extract_local(gathered_logits, accelerator.num_processes)

    pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
    # check if the logits are correct, extract argmax id
    # compare the predicted_ids with the labels
    pred_text = tokenizer.decode(pred.squeeze().tolist())
    answer_text = tokenizer.decode(answer_ids.squeeze().tolist())
    correct = pred_text.replace(" ", "").lower() == answer_text.replace(" ", "").lower()
    if accelerator.is_main_process:
        print(
            "Predicted: ",
            pred_text,
            "Answer: ",
            answer_text,
        )
        # print id as well
        print(
            "Predicted: ",
            pred.squeeze().tolist(),
            "Answer: ",
            answer_ids.squeeze().tolist(),
        )
    return int(correct)


def load_video(args, video_uid):
    video_embeddings = torch.load(f"{args.video_embeddings_dir}/{video_uid}.pt").to(torch.bfloat16)
    return video_embeddings


def load_text_ids(str, tokenizer, replace_double_newline=False):
    token_ids = safe_tokenize(tokenizer, str)

    def replace_double_newline_func(token_ids):
        double_newline_loc = (token_ids == 271).nonzero()[:, 1]
        double_newline_loc += torch.arange(len(double_newline_loc))
        if len(double_newline_loc) > 0:
            for loc in double_newline_loc:
                token_ids = torch.cat(
                    [
                        token_ids[:, :loc],
                        torch.tensor([[198, 198]]),
                        token_ids[:, loc + 1 :],
                    ],
                    dim=1,
                )
        return token_ids

    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    return token_ids


def get_model_name(model):
    model_split = [name for name in model.split("/") if len(name) > 0]
    model_name = f"{model_split[-2]}_{model_split[-1]}"
    return model_name


def load_results(results_dir):
    results = []
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if "json" in file:
                    print("file", file)
                    results.append(json.load(open(os.path.join(root, file))))
    else:
        os.system("mkdir -p %s" % results_dir)
    return results

def load_instances(ego4d_dir):
    ego4d_file = os.path.join(ego4d_dir, "ego4d.json")
    with open(ego4d_file) as f:
        data = json.load(f)
    ego4d_val = os.path.join(ego4d_dir, "v2", 'annotations', "nlq_val.json")
    with open(ego4d_val) as f:
        val = json.load(f)
    videos_metadata = {x['video_uid']: x for x in data['videos']}

    # Reorder_inputs
    instances = []
    for d in val['videos']:
        video_id = d['video_uid']
        video_metadata = videos_metadata[video_id]
        video_duration = video_metadata['duration_sec']
        for clip in d['clips']:
            clip_id = clip['clip_uid']
            for annotation in clip['annotations']:
                annotation_id = annotation['annotation_uid']
                for i, sample in enumerate(annotation['language_queries']):
                    instances.append({
                        "id": f"{annotation_id}_{i}",
                        "video_uid": video_id,
                        "duration_sec": video_duration,
                        'clip_start_sec': sample.get('clip_start_sec'),
                        'clip_end_sec': sample.get('clip_end_sec'),
                        'video_start_frame': sample.get('video_start_frame'),
                        'video_end_frame': sample.get('video_end_frame'),
                        'template': sample.get('template'),
                        'query': sample.get('query')
                    })
    return instances


def inference(args):

    # Load data
    instances = load_instances(args.ego4d_dir)
    instances = instances[:2]

    # Compute video_embeddings
    video_processor = VideoProcessor(
        args.model, args.video_dir, args.frames_batch_size, 
        args.max_frame_num, args.pooling_size, args.add_newline_token, args.video_embeddings_dir
        )
    video_uids = list(set([x["video_uid"] for x in instances]))
    video_processor.precompute_video_features(video_uids)
    del video_processor
    torch.cuda.empty_cache()
    gc.collect()

    model_path = args.model
    model_name = get_model_name_from_path(model_path)
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    tokenizer, model, _, _ = load_pretrained_model(
            model_path, model_name, None, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2", **kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    # remember to remove <s>
    
    prompt = prompt_templates[args.prompt_template]
    preprompt_token_ids = load_text_ids(
        prompt["preprompt"], tokenizer, args.replace_double_newline
    )
    postprompt_token_ids = load_text_ids(
        prompt["postprompt"], tokenizer, args.replace_double_newline
    )

    question_token_ids_bank = []
    for instance in instances:
        question = instance["query"]
        question_token_ids = load_text_ids(question, tokenizer, args.replace_double_newline)
        question_token_ids_bank.append(question_token_ids)

    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()

    generation_config = model.default_generation_config

    predictions = []
    for question_token_ids, instance in zip(question_token_ids_bank, instances):
        question = instance["query"]
        video_uid = instance["video_uid"]
        video = Video(os.path.join(args.video_dir, f"{video_uid}.mp4"))
        conversation = [{"from": "human", "value": [video, question]}]
        media_config = defaultdict(dict)
        input_ids = tokenize_conversation(conversation, tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)
        video_embeddings = load_video(args, video_uid)
        media = (video_embeddings.view(-1, video_embeddings.shape[-1]).unsqueeze(0))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            print("Generation failed with sampling, retrying with greedy decoding.")
            generation_config.do_sample = False
            output_ids = model.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # gc.collect()
        # torch.cuda.empty_cache()
        if accelerator.is_main_process:
            instance[f'{model_name}_response'] = response
            predictions.append(instance)

    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        with open(f"{args.output_path}/{model_name}/responses.jsonl", "w") as f:
            for item in predictions:
                f.write(json.dumps(item) + "\n")
    return predictions, accelerator


def main(args):
    inference(args)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="Efficient-Large-Model/qwen2-7b-longvila-1M")
    args.add_argument("--output_path", type=str, default="vision_ego4D")
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument("--ego4d_dir", type=str, default="/nethome/che321/flash/datasets/Ego4D")
    args.add_argument(
        "--video_embeddings_dir",
        type=str,
        default="data/video_embeddings/LongVILA-7B-1M",
    )
    args.add_argument("--prompt_template", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--video_dir", type=str, default="/nethome/che321/flash/datasets/Ego4D/v2/nlq_videos/full_scale")
    args.add_argument("--frames_batch_size", type=int, default=32)
    args.add_argument("--max_frame_num", type=int, default=7200)
    args.add_argument("--pooling_size", type=int, default=0)
    args.add_argument("--add_newline_token", action="store_true")

    main(args.parse_args())


