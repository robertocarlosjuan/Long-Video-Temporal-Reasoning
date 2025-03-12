#!/bin/bash
#SBATCH --job-name=lvila_ego4d
#SBATCH --output=lvila_ego4d1.out
#SBATCH --error=lvila_ego4d1.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --qos="short"
#SBATCH --gpus-per-node="a40:2"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate vila

MODEL_NAME="LongVILA-7B-1M"
MODEL_PATH="Efficient-Large-Model/qwen2-7b-longvila-1M"
VIDEO_DIR="/nethome/che321/flash/datasets/Ego4D/v2/nlq_videos/full_scale/"
prompt_template=qwen2
max_frame_num=100

eval_path=llava/eval/ego4d_vila
video_embeddings_dir=data/video_embeddings/$MODEL_NAME
output_path=results
mkdir -p $video_embeddings_dir
mkdir -p $output_path
video_embeddings_dir=$(realpath $video_embeddings_dir)
output_path=$(realpath $output_path)
echo "$video_embeddings_dir"
echo "$output_path"

cd ~/flash/VILA
# python $eval_path/produce_video_embedding.py --model $MODEL_PATH --output_dir $video_embeddings_dir --sampled_frames_num $max_frame_num --pooling_size 0 --video_path $VIDEO_PATH
accelerate launch --num_processes 2 --config_file  scripts/deepspeed_inference.yaml  --main_process_port 6000 $eval_path/eval_vision_ego4d.py \
    --model  $MODEL_PATH \
    --video_embeddings_dir $video_embeddings_dir \
    --prompt_template $prompt_template \
    --output_path $output_path \
    --video_dir $VIDEO_DIR \
    --max_frame_num $max_frame_num \
    --pooling_size 0
