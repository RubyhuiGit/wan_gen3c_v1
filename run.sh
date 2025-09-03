# train
accelerate launch examples/wanvideo/model_training/train_gen3c.py \
  --dataset_base_path /root/autodl-tmp/10_K \                     # 数据的根目录
  --dataset_metadata_path /root/autodl-tmp/10_K/metadata.json \   # 参考readme.md
  --height 480 \              # 暂时只测试了H=480的图，宽度不受限
  --width 832 \
  --dataset_repeat 100 \      # 每个数据重复的次数  下面的模型要确保在最完成的models/PAI下面能找到
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-InP:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-InP:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/root/autodl-tmp/diff_res" \     # 保存路径
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --find_unused_parameters \
  --video_sample_stride 1 \                 # 这个是图片的采样间隔
  --video_sample_n_frames 21 \              # 要采样的视频数量，这个数必须除以4后，余数=1
  --extra_inputs "input_image" \
  --cache_index 0 20                        # 要在21帧的范围内设置，回头写个check函数

# test （里面要改一下）
# python examples/wanvideo/model_training/test_gen3c.py