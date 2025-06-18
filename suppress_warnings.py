#!/usr/bin/env python
import warnings
import os

# 设置环境变量来忽略警告
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

# 忽略特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

print("警告已抑制，现在运行您的训练命令")

# 获取命令行参数
import sys
if len(sys.argv) > 1:
    # 构建命令
    command = ' '.join(sys.argv[1:])
    # 执行命令
    os.system(command)
else:
    print("使用方法: python suppress_warnings.py <您的训练命令>")
    print("例如: python suppress_warnings.py CUDA_VISIBLE_DEVICES=5 python train_net.py --config-file configs/san_clip_vit_res4_coco.yaml --num-gpus 1 OUTPUT_DIR ./output/train_vit_14_1") 