# # train
# accelerate launch examples/wanvideo/model_training/train_gen3c.py \
#   --dataset_base_path /root/autodl-tmp/10_K \
#   --dataset_metadata_path /root/autodl-tmp/10_K/metadata.json \
#   --height 480 \
#   --width 832 \
#   --dataset_repeat 100 \
#   --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-InP:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-InP:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "/root/autodl-tmp/diff_res" \
#   --lora_base_model "dit" \
#   --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
#   --lora_rank 32 \
#   --find_unused_parameters \
#   --video_sample_stride 1 \
#   --video_sample_n_frames 21 \
#   --extra_inputs "input_image,end_image" \
#   --cache_index 0 20

# test
python examples/wanvideo/model_training/test_gen3c.py