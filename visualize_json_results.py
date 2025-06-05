#!/usr/bin/env python  # 指定脚本解释器为python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved  # 版权声明

import argparse  # 导入argparse模块，用于解析命令行参数
import json  # 导入json模块，用于处理JSON数据
import numpy as np  # 导入NumPy库，用于数值操作
import os  # 导入os模块，用于操作系统相关功能
from collections import defaultdict  # 从collections模块导入defaultdict，用于创建默认值的字典
import cv2  # 导入OpenCV库，用于图像处理
import tqdm  # 导入tqdm库，用于显示进度条
from fvcore.common.file_io import PathManager  # 从fvcore.common.file_io导入PathManager，用于文件路径管理
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
from detectron2.data import DatasetCatalog, MetadataCatalog  # 从detectron2.data导入DatasetCatalog和MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances  # 从detectron2.structures导入Boxes, BoxMode, Instances (Boxes和BoxMode在此脚本中未使用)
from detectron2.utils.logger import setup_logger  # 从detectron2.utils.logger导入setup_logger，用于设置日志记录器
from detectron2.utils.visualizer import Visualizer, GenericMask  # 从detectron2.utils.visualizer导入Visualizer和GenericMask，用于可视化
import sys  # 导入sys模块 (在此脚本中未使用)

import san  # 导入san模块 (在此脚本中具体用途未明确显示，可能是为了注册自定义内容)


def create_instances(predictions, image_size, ignore_label=255):  # 定义创建实例分割图的函数
    ret = Instances(image_size)  # 创建一个空的Instances对象

    labels = np.asarray(  # 将预测的类别ID转换为NumPy数组
        [dataset_id_map(predictions[i]["category_id"]) for i in range(len(predictions))]
    )
    ret.pred_classes = labels  # 设置Instances对象的预测类别
    ret.pred_masks = [  # 创建GenericMask对象列表作为预测掩码
        GenericMask(predictions[i]["segmentation"], *image_size)
        for i in range(len(predictions))
    ]
    # convert instance to sem_seg map
    sem_seg = np.ones(image_size[:2], dtype=np.uint16) * ignore_label  # 创建一个用忽略标签初始化的语义分割图
    for mask, label in zip(ret.pred_masks, ret.pred_classes):  # 遍历预测的掩码和类别
        sem_seg[mask.mask == 1] = label  # 将掩码对应的区域填充为类别标签
    return sem_seg  # 返回语义分割图


if __name__ == "__main__":  # 如果作为主脚本运行
    parser = argparse.ArgumentParser(  # 创建命令行参数解析器
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument(  # 添加输入JSON文件参数
        "--input", required=True, help="JSON file produced by the model"
    )
    parser.add_argument("--output", required=True, help="output directory")  # 添加输出目录参数
    parser.add_argument(  # 添加数据集名称参数
        "--dataset", help="name of the dataset", default="coco_2017_val"
    )
    parser.add_argument(  # 添加置信度阈值参数 (在此脚本中未使用)
        "--conf-threshold", default=0.5, type=float, help="confidence threshold"
    )
    args = parser.parse_args()  # 解析命令行参数

    logger = setup_logger()  # 设置日志记录器

    with PathManager.open(args.input, "r") as f:  # 打开输入的JSON文件
        predictions = json.load(f)  # 加载JSON数据

    pred_by_image = defaultdict(list)  # 创建一个默认值为列表的字典，用于按图像文件名组织预测结果
    for p in predictions:  # 遍历所有预测结果
        pred_by_image[p["file_name"]].append(p)  # 将预测结果按文件名添加到字典中

    dicts = list(DatasetCatalog.get(args.dataset))  # 获取指定数据集的字典列表
    metadata = MetadataCatalog.get(args.dataset)  # 获取指定数据集的元数据
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):  # 如果元数据中有thing_dataset_id_to_contiguous_id属性 (COCO数据集)

        def dataset_id_map(ds_id):  # 定义数据集ID到连续ID的映射函数
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:  # 如果数据集名称包含"lvis" (LVIS数据集)
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):  # 定义LVIS数据集ID的映射函数 (LVIS ID从1开始)
            return ds_id - 1

    elif "sem_seg" in args.dataset:  # 如果数据集名称包含"sem_seg" (通用语义分割数据集)

        def dataset_id_map(ds_id):  # 定义语义分割数据集ID的映射函数 (通常ID已经是连续的)
            return ds_id

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))  # 如果数据集不支持，则抛出错误

    os.makedirs(args.output, exist_ok=True)  # 创建输出目录，如果目录已存在则不报错

    for dic in tqdm.tqdm(dicts):  # 遍历数据集字典列表，并使用tqdm显示进度条
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]  # 读取图像文件，并从BGR转换为RGB
        basename = os.path.basename(dic["file_name"])  # 获取图像文件名
        if dic["file_name"] in pred_by_image:  # 如果当前图像有对应的预测结果
            pred = create_instances(  # 创建预测的语义分割图
                pred_by_image[dic["file_name"]],
                img.shape[:2],  # 图像的高和宽
                ignore_label=metadata.ignore_label,  # 忽略标签
            )

            vis = Visualizer(img, metadata)  # 创建Visualizer对象用于绘制预测结果
            vis_pred = vis.draw_sem_seg(pred).get_image()  # 绘制预测的语义分割图并获取图像
            # import pdb
            # pdb.set_trace() # 调试断点 (已注释)
            vis = Visualizer(img, metadata)  # 重新创建Visualizer对象用于绘制真实标签 (确保颜色映射一致或重新开始)
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:  # 打开真实标签的语义分割文件
                sem_seg_gt = Image.open(f)  # 使用PIL打开图像
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint16")  # 将PIL图像转换为NumPy数组
            vis_gt = vis.draw_sem_seg(sem_seg_gt).get_image()  # 绘制真实标签的语义分割图并获取图像
            # reisze pred and gt to the same height
            ratio = vis_gt.shape[0] / 512  # 计算高度缩放比例，目标高度为512
            tgt_w = int(vis_pred.shape[1] / ratio)  # 根据比例计算目标宽度
            vis_pred = cv2.resize(vis_pred, (tgt_w,512))  # 缩放预测可视化结果
            vis_gt = cv2.resize(vis_gt, (tgt_w,512))  # 缩放真实标签可视化结果
            img_resized = cv2.resize(img, (tgt_w,512)) # 缩放原始图像
            # build grid view
            blank_int = 255 * np.ones((vis_gt.shape[0], 10, 3), dtype=np.uint8)  # 创建一个白色空白分隔条
            concat = np.concatenate(  # 将原始图像、预测可视化、真实标签可视化拼接在一起
                (img_resized, blank_int, vis_pred, blank_int, vis_gt), axis=1
            )
            cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])  # 保存拼接后的图像 (从RGB转回BGR以供cv2保存)