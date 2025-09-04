
import torch
import numpy as np
import os
import json
import random
import torchvision.transforms as transforms
from PIL import Image
from .data_utils.dataset_10k_parse import Dataset10KParse

class Dataset10K(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root=None,
        video_sample_stride=4, video_sample_n_frames=16,
        cache_index=None,
        repeat_num=100,
        text_drop_ratio=0.1,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        sample_size=None,
    ):  
        self.data_root = data_root
        json_file_path = os.path.join(data_root, "metadata.json")
        self.dataset = json.load(open(json_file_path))                           # 加载json文件

        print("Load Dataset From:", json_file_path)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")                         # 数据集大小

        self.text_drop_ratio = text_drop_ratio

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride        # 采样间隔 3
        self.video_sample_n_frames  = video_sample_n_frames      # 期待的帧数

        self.sample_size = sample_size
        self.cache_index = cache_index

        self.repeat = repeat_num

    def get_batch(self, idx):
        json_data_info = self.dataset[idx % len(self.dataset)]
        dataset_dir = os.path.join(self.data_root, json_data_info['file_path'])

        scene_json_file = os.path.join(dataset_dir, "_scene_meta_backup.json")
        if not os.path.exists(scene_json_file):
            raise ValueError(f"Scene Json File Not exsit!")
        scene_info = json.load(open(scene_json_file, 'r'))
        scene_frames_info = scene_info["frames"]

        video_len = len(scene_frames_info)        # 视频长度

        # 期待帧数 self.video_sample_n_frames   (多读取2帧)
        # 现有视频有效长度 int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start),  有效帧数 = 有效长度 / 采样间隔
        # 期待帧数和有效长度直接选最小（考虑了采样间隔）
        min_sample_n_frames = min(
            self.video_sample_n_frames, 
            int(video_len * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
        )
        if min_sample_n_frames == 0:
            raise ValueError(f"No Frames in video.")

        video_length = int(self.video_length_drop_end * video_len)    # 视频有效帧的末尾位置    324
        clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)    # 最多采样帧数 * 时间跨度 = 最远采样长度

        min_sample_n_frames = min_sample_n_frames + (1 - (min_sample_n_frames % 4)) % 4       # 保证min_sample_n_frames是4的倍数 + 1

        # 随机选择一个起始帧 & 生成要抽取帧的索引
        start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

        dataset_10k_parse_tools = Dataset10KParse(dataset_dir)
        data_info = dataset_10k_parse_tools.parse(batch_index, True)
        del dataset_10k_parse_tools

        frame_info = {}
        if data_info == None:
            print("Parse File Failed:", dataset_dir)
            raise ValueError(f"Parse Error, Check you dataset {dataset_dir}")
        else:
            frame_info["rgb_frames"] = data_info["frames"]
            frame_info["w2cs"] = data_info["w2c"]
            frame_info["intrinsics"] = data_info["intrinsics"]
            frame_info["depths"] = data_info["depth"]

        text = json_data_info.get('text', '')
        if random.random() < self.text_drop_ratio:
            text = ''

        valid_cache_index = []
        for cache_i in self.cache_index:
            if cache_i >= frame_info["rgb_frames"].shape[0]:
                valid_cache_index.append(frame_info["rgb_frames"].shape[0]-1)
            elif cache_i < 0:
                valid_cache_index.append(0)
            else:
                valid_cache_index.append(cache_i)
        frame_info["cache_index"] = valid_cache_index

        return frame_info, text, 'video', dataset_dir

    def __len__(self):
        return self.length * self.repeat

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                data_info, text, file_type, dataset_dir = self.get_batch(idx)
                sample = data_info
                sample["text"] = text
                sample["data_type"] = file_type
                sample["idx"] = idx
                sample["file_name"] = dataset_dir
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)
        return sample


class Dataset10KTestInfo():
    def __init__(
        self,
        data_root=None,
        cache_index=None,
        sample_frames_num=81,
        max_frame_num=300
    ):  
        self.data_root = data_root
        self.cache_index = cache_index
        self.sample_frames_num = sample_frames_num
        self.max_frame_num = max_frame_num

    def get_data_dict(self, select_index=None, text=""):
        dataset_10k_parse_tools = Dataset10KParse(self.data_root)
        select_index = np.linspace(0, self.max_frame_num, self.sample_frames_num, dtype=int)
        data_info = dataset_10k_parse_tools.parse(select_index, True)
        del dataset_10k_parse_tools

        frame_info = {}
        if data_info == None:
            print("Parse File Failed:", dataset_dir)
            raise ValueError(f"Parse Error, Check you dataset {dataset_dir}")
        else:
            # data_info["frames"]         # N, H, W, 3
            # data_info["w2c"]            # N, 4, 4
            # data_info["intrinsics"]     # N, 4, 3
            # data_info["depth"]          # N, H, W

            frame_info["target_w2c"] = data_info["w2c"]
            frame_info["target_intrinsic"] = data_info["intrinsics"]

            cache_frames = [data_info["frames"][i] for i in self.cache_index]
            cache_w2cs = [data_info["w2c"][i] for i in self.cache_index]
            cache_intrinsics = [data_info["intrinsics"][i] for i in self.cache_index]
            cache_depths = [data_info["depth"][i] for i in self.cache_index]

            frame_info["cache_frames"] = np.stack(cache_frames, axis=0)
            frame_info["cache_w2cs"] = np.stack(cache_w2cs, axis=0)
            frame_info["cache_intrinsics"] = np.stack(cache_intrinsics, axis=0)
            frame_info["cache_depths"] = np.stack(cache_depths, axis=0)
            frame_info["first_frame"] = data_info["frames"][0]

            N, H, W, _ = data_info["frames"].shape
            frame_info["num_frames"] = N
            frame_info["height"] = H
            frame_info["width"] = W

        frame_info["cache_index"] = self.cache_index
        return frame_info

