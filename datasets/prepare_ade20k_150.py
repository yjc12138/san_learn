#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image


# def convert(input, output):
#     img = np.asarray(Image.open(input)) # 读取输入图像并转换为numpy数组
#     assert img.dtype == np.uint8 # 确保图像是8位无符号整数格式
#     img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
#     Image.fromarray(img).save(output) # 保存处理后的图像


# if __name__ == "__main__":
#     dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016"
#     for name in ["training", "validation"]:
#         annotation_dir = dataset_dir / "annotations" / name
#         output_dir = dataset_dir / "annotations_detectron2" / name
#         output_dir.mkdir(parents=True, exist_ok=True)
#         for file in tqdm.tqdm(list(annotation_dir.iterdir())):
#             output_file = output_dir / file.name
#             convert(file, output_file)

def convert(input, output):
    img = np.asarray(Image.open(input)) # 读取输入图像并转换为numpy数组
    assert img.dtype == np.uint8 # 确保图像是8位无符号整数格式
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output) # 保存处理后的图像


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016"
    for name in ["training", "validation"]:
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)