

# 环境
1、参考README.md安装一下
2、requirements.txt 
3、要是transformers报错， pip install -U git+https://github.com/huggingface/transformers  
   我用的这个版本   transformers-4.57.0.dev0

# train & test
1、修改参数，运行run.sh，开始训练
2、python examples/wanvideo/model_training/test_gen3c.py 开始test, test_gen3c.py的参数也要修改一下

# metadata.json 格式如下
[
    {
      "file_path": "数据目录的相对路径",
      "text": "随便写一个",
      "type": "video"
    },
    {
      "file_path": "10K_2",
      "text": "Normal video",
      "type": "video"
    },
    {
      "file_path": "10K_3",
      "text": "Normal video",
      "type": "video"
    }
]