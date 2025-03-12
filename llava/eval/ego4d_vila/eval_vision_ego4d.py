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
import torch.nn.functional as F

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

# def inference_forward(accelerator, model, input_embeds, pad_id, tokenizer, max_new_tokens=50):
#     # Get the length of the prompt and the answer (in tokens)
#     prompt_length = input_embeds.shape[1]
    
#     # Pad input_embeds to a multiple of accelerator.num_processes * 2
#     total_width = accelerator.num_processes * 2
#     remainder = input_embeds.shape[1] % total_width
#     if remainder:
#         pad_length = total_width - remainder
#         pad_tensor = (
#             torch.tensor([pad_id] * pad_length)
#             .unsqueeze(0)
#             .unsqueeze(-1)
#             .expand(input_embeds.shape[0], pad_length, input_embeds.shape[-1])
#         )
#         input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    
#     # Create position ids for the entire padded input
#     position_ids = torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    
#     # Prepare attention inputs (assuming this function handles the local sharding)
#     prepared = prepare_zigzag_ring_attn_inputs(
#         input_embeds,
#         position_ids,
#         None,
#         accelerator.process_index,
#         accelerator.num_processes,
#         accelerator.device,
#     )
#     local_input_embeds = prepared["local_input_ids"]
#     local_position_ids = prepared["local_position_ids"]
    
#     # Generate answer tokens using model.generate
#     with torch.inference_mode():
#         generated_ids = model.generate(
#             inputs_embeds=local_input_embeds,
#             max_new_tokens=max_new_tokens,  # generate as many tokens as in answer_ids
#             position_ids=local_position_ids,
#             use_cache=False,
#             pad_token_id=pad_id
#         )
    
#     # The accelerator might be sharded across processes. The following helper undoes
#     # the splitting done during preparation.
#     def undo_extract_local(gathered_value, world_size, dim=1):
#         value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
#         reordered_chunks = [None] * (2 * world_size)
#         for i in range(world_size):
#             reordered_chunks[i] = value_chunks[i * 2]
#             reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
#         return torch.cat(reordered_chunks, dim=dim)
    
#     # Gather generation results from all processes
#     gathered_generated_ids = accelerator.gather(generated_ids.squeeze(0)).unsqueeze(0)
#     generated_ids = undo_extract_local(gathered_generated_ids, accelerator.num_processes)
    
#     # Slice off the prompt tokens; we only compare the generated part
#     generated_answer_ids = generated_ids[:, prompt_length:prompt_length+labels_length]
    
#     # Decode to text and compare to the expected answer
#     pred_text = tokenizer.decode(generated_answer_ids.squeeze().tolist())
    
#     if accelerator.is_main_process:
#         print("Predicted:", pred_text)
#     return pred_text


def sample_next_token(logits, top_k=50, temperature=1.0):
    """
    Samples a token from the logits using top-k sampling and temperature scaling.
    
    Args:
        logits (torch.Tensor): Logits of shape [batch, seq_length, vocab_size].
        top_k (int): The number of top tokens to consider for sampling.
        temperature (float): Temperature for scaling logits.
    
    Returns:
        torch.Tensor: Sampled token id(s) of shape [batch, 1].
    """
    # Consider only the logits for the last token in the sequence.
    logits = logits[:, -1, :]  # shape: [batch, vocab_size]
    logits = logits / temperature

    # Get the top_k tokens and their indices.
    top_k = min(top_k, logits.size(-1))
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)

    # Create a probability distribution from the top k logits.
    probabilities = F.softmax(top_logits, dim=-1)
    
    # Sample from the distribution.
    next_token = torch.multinomial(probabilities, num_samples=1)  # shape: [batch, 1]
    
    # Map back to original token ids.
    next_token_id = top_indices.gather(-1, next_token)
    return next_token_id

def inference_forward(accelerator, model, input_embeds, pad_id, tokenizer, max_new_tokens=50, top_k=50, temperature=1.0):
    """
    Generates text token-by-token given a prompt in the form of input embeddings.
    
    Args:
        accelerator: An accelerator object providing process info and methods.
        model: The model to generate from.
        input_embeds: The prompt embeddings (shape: [batch, seq_length, hidden_dim]).
        pad_id: The pad token id used for padding.
        tokenizer: A tokenizer with a decode method and an eos_token_id attribute.
        max_new_tokens: Maximum number of new tokens to generate.
        
    Returns:
        generated_text: The generated text decoded from token ids.
    """
    
    def undo_extract_local(gathered_value, world_size, dim=1):
        # Reorder chunks gathered from distributed inference.
        value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
        reordered_chunks = [None] * (2 * world_size)
        for i in range(world_size):
            reordered_chunks[i] = value_chunks[i * 2]
            reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
        return torch.cat(reordered_chunks, dim=dim)
    
    model.eval()
    generated_ids = []  # to store generated token ids
    # We assume input_embeds is already on the proper device.
    input_embeds = input_embeds.to(accelerator.device)
    for tokens_i in range(max_new_tokens):
        # Ensure the sequence length is padded to a multiple of (accelerator.num_processes * 2)
        print("INPUT EMBEDDINGS: ", input_embeds)
        print(f"{tokens_i}. input_embeds: {input_embeds.shape}")
        seq_length = input_embeds.shape[1]
        pad_multiple = accelerator.num_processes * 2
        remainder = seq_length % pad_multiple
        created_tensor = False
        if remainder != 0:
            pad_length = pad_multiple - remainder
            pad_tensor = (
                torch.tensor([pad_id] * pad_length, device=input_embeds.device)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(input_embeds.shape[0], pad_length, input_embeds.shape[-1])
            )
            input_padded_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
            created_tensor = True
            print(f"{tokens_i}. Padding applied: pad_length {pad_length}")
        else:
            input_padded_embeds = input_embeds
            print(f"{tokens_i}. No padding needed.")
        print(f"{tokens_i}. input_padded_embeds: {input_padded_embeds.shape}")
        
        # Create position ids for the (padded) sequence.
        position_ids = (
            torch.arange(input_padded_embeds.shape[1], device=input_padded_embeds.device)
            .unsqueeze(0)
            .expand(input_padded_embeds.shape[0], -1)
        )
        print(f"{tokens_i}. position_ids shape: {position_ids.shape}")
        
        # Prepare inputs for the zigzag ring attention.
        with torch.inference_mode():
            prepared = prepare_zigzag_ring_attn_inputs(
                input_padded_embeds,
                position_ids,
                None,
                accelerator.process_index,
                accelerator.num_processes,
                accelerator.device,
            )
            local_input_embeds = prepared["local_input_ids"]
            local_position_ids = prepared["local_position_ids"]
        
            # Forward pass in inference mode.
            logits = model(
                inputs_embeds=local_input_embeds,
                position_ids=local_position_ids,
                use_cache=False,
            ).logits

            pred = logits.argmax(dim=-1)
        
        # Gather predictions from all processes.
        gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
        pred = undo_extract_local(gathered_logits, accelerator.num_processes)
        next_token_id = pred[:, -1].unsqueeze(1)  # shape: [batch, 1]

        # gathered_logits = accelerator.gather(logits)
        # logits = undo_extract_local(gathered_logits, accelerator.num_processes)
        # # pred = logits.argmax(dim=-1)
        # next_token_id = sample_next_token(logits, top_k=top_k, temperature=temperature)
        
        # The next token is taken as the last token in the sequence.
        generated_ids.append(next_token_id)
        print(f"{tokens_i}. Next token id (raw): {next_token_id.tolist()}")
        temp_generated_ids = torch.cat(generated_ids, dim=1)
        temp_decoded = tokenizer.decode(temp_generated_ids.squeeze().tolist())
        print(f"{tokens_i}. Partial generated text: {temp_decoded}")
        
        # If EOS token is generated, stop early.
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"{tokens_i}. EOS token generated. Stopping.")
            break
        
        # Get the embedding of the newly generated token.
        next_token_embed = model.model.embed_tokens(next_token_id)
        print(f"{tokens_i}. next_token_embed shape: {next_token_embed.shape}")
        
        # Append the new token embedding to the current input embeddings.
        input_embeds = torch.cat([input_embeds, next_token_embed], dim=1)
        print(f"{tokens_i}. Updated input_embeds shape: {input_embeds.shape}")
        if created_tensor:
            del input_padded_embeds
            gc.collect()
            torch.cuda.empty_cache()
    
    # Concatenate generated token ids and decode them.
    generated_ids = torch.cat(generated_ids, dim=1)  # shape: [batch, num_generated]
    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    
    if accelerator.is_main_process:
        print("Generated text:", generated_text)
    
    return generated_text


def load_video(args, video_uid):
    video_embeddings = torch.load(f"{args.video_embeddings_dir}/{video_uid}.pt").to(torch.bfloat16)
    return video_embeddings


def load_text_embeddings(str, tokenizer, model, accelerator, replace_double_newline=False):
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
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)


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
    model = args.model

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

    config = AutoConfig.from_pretrained(args.model)
    cfgs = get_model_config(config)
    llm_path = cfgs[0]

    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    if "qwen2" in args.model.lower() or "longva" in args.model.lower():
        model = Qwen2ForCausalLM_RingAttn.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            **kwargs,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    
    prompt = prompt_templates[args.prompt_template]
    preprompt_embeddings = load_text_embeddings(
        prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline
    )
    postprompt_embeddings = load_text_embeddings(
        prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline
    )

    question_embedding_bank = []
    for instance in instances:
        question = instance["query"]
        question_embedding = load_text_embeddings(question, tokenizer, model, accelerator)
        question_embedding_bank.append(question_embedding)

    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()

    model_name = get_model_name(args.model)

    # generation_config = default_generation_config(tokenizer)

    predictions = []
    for question_embedding, instance in zip(question_embedding_bank, instances):
        question = instance["query"]
        video_uid = instance["video_uid"]
        video_embeddings = load_video(args, video_uid)
        input_frames = (video_embeddings.view(-1, video_embeddings.shape[-1]).unsqueeze(0))
        input_emebds = torch.cat(
            [
                preprompt_embeddings,
                input_frames,
                question_embedding,
                postprompt_embeddings,
            ],
            dim=1,
        )
        response = inference_forward(
            accelerator,
            model,
            input_emebds,
            tokenizer.pad_token_id,
            tokenizer,
            max_new_tokens=50
        )
        gc.collect()
        torch.cuda.empty_cache()
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


