import torch, os, json
import numpy as np
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_3dcache_pipeline import WanVideo3DCachePipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
from data.dataset_10k import Dataset10K
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideo3DCachePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(self.pipe, lora_base_model, model)
        
        self.pipe.init_cache_adaptor()
    
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):

        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["text"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["rgb_frames"],          # 81, 480, 640, 3
            "input_w2c": data["w2cs"],                  # 81, 4, 4
            "input_intrinsics": data["intrinsics"],     # 81, 3, 3
            "input_depth": data["depths"],              # 81, 480, 640
            "height": data["rgb_frames"][0].shape[0],
            "width": data["rgb_frames"][0].shape[1],
            "num_frames": len(data["rgb_frames"]),
            "cache_index": data["cache_index"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["rgb_frames"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["rgb_frames"][-1]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            # print(unit, unit.take_over, unit.seperate_cfg)
            # WanVideoUnit_ShapeChecker                 处理以后 h=480, w=640, num_frames=81  填充位置 inputs_shared
            # WanVideoUnit_3DCacheRender                render_control_latents # torch.Size([1, 64, 21, 60, 80])
            # WanVideoUnit_NoiseInitializer             噪声 torch.Size([1, 16, 21, 60, 80]))  填充位置 inputs_shared
            # WanVideoUnit_PromptEmbedder               prompt torch.Size([1, 512, 4096])  填充 inputs_posi
            # WanVideoUnit_S2V                          不进入
            # WanVideoUnit_InputVideoEmbedder           对视频进行vae编码   填充位置 inputs_shared   noise 1, 16, 21, 60, 80   input_latents 1, 16, 21, 60, 80    
            # WanVideoUnit_CacheImageEmbedderVAE        对指定的cache frame vae处理   torch.Size([1, 20, 21, 60, 80])   特征20 时间21 填充位置 inputs_shared
            # WanVideoUnit_ImageEmbedderCLIP            只是首帧编码     # torch.Size([1, 257, 1280])   填充位置 inputs_shared
            # 下面的unit不用管
            # WanVideoUnit_ImageEmbedderFused           不进入
            # WanVideoUnit_FunControl  WanVideoUnit_FunReference  WanVideoUnit_FunCameraControl  都不进入
            # WanVideoUnit_SpeedControl                 不进入
            # WanVideoUnit_VACE                         不进入
            # WanVideoUnit_UnifiedSequenceParallel      use_unified_sequence_parallel 参数设置
            # WanVideoUnit_TeaCache                     tea_cache相关参数
            # WanVideoUnit_CfgMerger                    不进入
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)        # 除了dit以外，其他都是None
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = Dataset10K(args.dataset_base_path,
                         args.video_sample_stride,
                         args.video_sample_n_frames,
                         args.cache_index,
                         args.dataset_repeat)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # if True:
    #     trainable_params = model.trainable_param_names()
    #     log_file = open("/root/train_params.json", 'w')
    #     for item in sorted(list(trainable_params)):
    #         print(item, file=log_file)                      # 打印设置了grad=True的参数    
    #     param_id_to_name_map = {id(p): name for name, p in model.named_parameters()} 
    #     params_in_optimizer_names = set()
    #     for param_group in optimizer.param_groups:
    #         for param in param_group['params']:
    #             param_id = id(param)
    #             if param_id in param_id_to_name_map:
    #                 params_in_optimizer_names.add(param_id_to_name_map[param_id])
    #         for name in sorted(list(params_in_optimizer_names)):
    #             print(name)

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
    )
