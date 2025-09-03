import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.models.utils import load_state_dict_from_safetensors
from diffsynth.pipelines.wan_video_3dcache_infer_pipeline import WanVideo3DCacheInferPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from data.dataset_10k import Dataset10KTestInfo

def split_params(file_path, delete_prefix="pipe.cache_adaptor."):
    state_dict = load_state_dict_from_safetensors(file_path)
    lora_state_dict = {}
    other_state_dict = {}
    for key, tensor in state_dict.items():
        if ".lora" in key:
            lora_state_dict[key] = tensor    
        if key.startswith(delete_prefix):
            other_state_dict[key.replace(delete_prefix, "", 1)] = tensor
    return lora_state_dict, other_state_dict

pipe = WanVideo3DCacheInferPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)

lora_params, adaptor_params = split_params("/root/autodl-tmp/diff_res/epoch-1.safetensors")
pipe.load_lora_from_dict(pipe.dit, lora_params)      # 加载lora文件   GeneralLoRALoader
pipe.enable_vram_management()
pipe.set_cache_adaptor(adaptor_params)

test_path = "/root/autodl-tmp/10_K/10K_1"
cache_index = [0, 20]     # 要指定用来创建3d cache的frame_index
test_info = Dataset10KTestInfo(test_path, cache_index, sample_frames_num=21, max_frame_num=21).get_data_dict()

# First and last frame to video
video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    dataset_info=test_info,
    seed=0, tiled=True
)
save_video(video, "/root/video_Wan2.1-Fun-V1.1-1.3B-InP.mp4", fps=15, quality=5)