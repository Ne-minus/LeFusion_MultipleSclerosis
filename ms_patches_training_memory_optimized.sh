#!/bin/bash

echo "🚀 Starting Memory-Optimized MS Patches Training"
echo "================================================"

dataset=ms
diffusion_img_size=64
diffusion_depth_size=32
diffusion_num_channels=1
batch_size=4
test_txt_dir=""
dataset_root_dir=MS_data/patches
train_num_steps=50001
cond_dim=16
results_folder=LeFusion/LeFusion_Model/MS_Patches_MemoryOptimized

gpu_id=0
num_workers=4

gradient_accumulate_every=8
amp=True
max_split_size_mb=512

echo "📁 Creating results directory: $results_folder"
mkdir -p $results_folder

if [ ! -d "$dataset_root_dir" ]; then
    echo "❌ Error: Data directory $dataset_root_dir not found!"
    exit 1
fi

echo "🔍 Checking available data..."
data_count=$(find $dataset_root_dir -name "*_image.nii.gz" | wc -l)
echo "📊 Found $data_count image files"

if [ $data_count -eq 0 ]; then
    echo "❌ Error: No image files found in $dataset_root_dir"
    exit 1
fi

echo "🖥️  Checking GPU availability and memory..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
    echo "✅ GPU detected"
else
    echo "⚠️  nvidia-smi not found, assuming GPU is available"
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$max_split_size_mb
export CUDA_LAUNCH_BLOCKING=1

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="ms_patches_memory_optimized_${timestamp}.log"

echo "📝 Starting training with log file: $log_file"
echo "🎯 Memory-optimized training configuration:"
echo "   - Dataset: $dataset"
echo "   - Data directory: $dataset_root_dir"
echo "   - Batch size: $batch_size (reduced for memory)"
echo "   - Gradient accumulation: $gradient_accumulate_every (effective batch: $((batch_size * gradient_accumulate_every)))"
echo "   - Mixed precision: $amp"
echo "   - Training steps: $train_num_steps"
echo "   - GPU: $gpu_id"
echo "   - Workers: $num_workers"
echo "   - Results folder: $results_folder"
echo "   - Max split size: ${max_split_size_mb}MB"
echo "================================================"

python LeFusion/train/train.py \
    dataset=$dataset \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.diffusion_num_channels=$diffusion_num_channels \
    dataset.test_txt_dir=$test_txt_dir \
    dataset.root_dir=$dataset_root_dir \
    model.train_num_steps=$train_num_steps \
    model.batch_size=$batch_size \
    model.cond_dim=$cond_dim \
    model.results_folder=$results_folder \
    model.save_and_sample_every=100 \
    model.ema_decay=0.995 \
    model.train_lr=1e-4 \
    model.gradient_accumulate_every=$gradient_accumulate_every \
    model.amp=$amp \
    model.num_sample_rows=1 \
    model.num_workers=$num_workers \
    model.gpus=$gpu_id \
    2>&1 | tee $log_file

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results saved in: $results_folder"
    echo "📝 Full log available in: $log_file"
else
    echo "❌ Training failed. Check the log file: $log_file"
    echo "💡 If still out of memory, try:"
    echo "   - Reduce batch_size to 2 or 1"
    echo "   - Increase gradient_accumulate_every to 16"
    echo "   - Reduce num_workers to 2"
    exit 1
fi
