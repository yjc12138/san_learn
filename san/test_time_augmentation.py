# 从mask2former仓库复制。
import copy  # 导入copy模块，用于深拷贝
import logging  # 导入日志模块
from itertools import count  # 导入count迭代器 (在此文件中未使用)

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from fvcore.transforms import HFlipTransform  # 从fvcore导入水平翻转变换
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 从PyTorch导入分布式数据并行

from detectron2.data.detection_utils import read_image  # 从detectron2导入读取图像函数
from detectron2.modeling import DatasetMapperTTA  # 从detectron2导入测试时增强的数据集映射器


__all__ = [
    "SemanticSegmentorWithTTA",  # 声明公开的类
]


class SemanticSegmentorWithTTA(nn.Module):
    """
    启用了测试时增强（TTA）的语义分割器。
    其 :meth:`__call__` 方法具有与 :meth:`SemanticSegmentor.forward` 相同的接口。
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        参数:
            cfg (CfgNode): 配置节点。
            model (SemanticSegmentor): 要应用TTA的SemanticSegmentor。
            tta_mapper (callable): 接受一个数据集字典并返回
                该数据集字典的增强版本列表。默认为
                `DatasetMapperTTA(cfg)`。
            batch_size (int): 将增强图像分批处理为此批次大小以进行推理。
        """
        super().__init__()  # 调用父类初始化方法
        if isinstance(model, DistributedDataParallel):  # 如果模型是分布式数据并行实例
            model = model.module  # 获取内部模型
        self.cfg = cfg.clone()  # 克隆配置节点

        self.model = model  # 存储模型

        if tta_mapper is None:  # 如果未提供TTA映射器
            tta_mapper = DatasetMapperTTA(cfg)  # 创建默认的TTA映射器
        self.tta_mapper = tta_mapper  # 存储TTA映射器
        self.batch_size = batch_size  # 存储批次大小

    def __call__(self, batched_inputs):
        """
        与 :meth:`SemanticSegmentor.forward` 相同的输入/输出格式
        """

        def _maybe_read_image(dataset_dict):
            # 可能读取图像的辅助函数
            ret = copy.copy(dataset_dict)  # 浅拷贝数据集字典
            if "image" not in ret:  # 如果字典中没有"image"键
                image = read_image(ret.pop("file_name"), self.model.input_format)  # 读取图像文件
                image = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                )  # CHW  # 转换为张量并调整维度
                ret["image"] = image  # 将图像存入字典
            if "height" not in ret and "width" not in ret:  # 如果字典中没有高度和宽度
                ret["height"] = image.shape[1]  # 从图像张量获取高度
                ret["width"] = image.shape[2]  # 从图像张量获取宽度
            return ret  # 返回处理后的字典

        processed_results = []  # 初始化处理结果列表
        for x in batched_inputs:  # 遍历批处理输入中的每个样本
            result = self._inference_one_image(_maybe_read_image(x))  # 对单个图像进行TTA推理
            processed_results.append(result)  # 将结果添加到列表
        return processed_results  # 返回处理后的结果列表

    def _inference_one_image(self, input):
        """
        对单个图像进行TTA推理。
        参数:
            input (dict): 一个数据集字典，其中"image"字段是一个CHW张量。
        返回:
            dict: 一个输出字典。
        """
        orig_shape = (input["height"], input["width"])  # 获取原始图像形状
        augmented_inputs, tfms = self._get_augmented_inputs(input)  # 获取增强后的输入和变换信息

        final_predictions = None  # 初始化最终预测
        count_predictions = 0  # 初始化预测计数器
        for input_aug, tfm in zip(augmented_inputs, tfms):  # 遍历每个增强后的输入和对应的变换
            count_predictions += 1  # 预测计数器加1
            with torch.no_grad():  # 在无梯度计算的上下文中执行
                if final_predictions is None:  # 如果是第一个增强版本的预测
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):  # 如果应用了水平翻转
                        final_predictions = (
                            self.model([input_aug])[0].pop("sem_seg").flip(dims=[2])
                        )  # 模型推理，获取语义分割结果，并翻转回来
                    else:
                        final_predictions = self.model([input_aug])[0].pop("sem_seg")  # 模型推理，获取语义分割结果
                else:  # 如果不是第一个增强版本的预测，则累加结果
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):  # 如果应用了水平翻转
                        final_predictions += (
                            self.model([input_aug])[0].pop("sem_seg").flip(dims=[2])
                        )  # 模型推理，获取语义分割结果，翻转回来，然后累加
                    else:
                        final_predictions += self.model([input_aug])[0].pop("sem_seg")  # 模型推理，获取语义分割结果，然后累加

        final_predictions = final_predictions / count_predictions  # 计算平均预测
        return {"sem_seg": final_predictions}  # 返回包含最终语义分割预测的字典

    def _get_augmented_inputs(self, input):
        # 获取增强后的输入
        augmented_inputs = self.tta_mapper(input)  # 使用TTA映射器生成增强输入
        tfms = [x.pop("transforms") for x in augmented_inputs]  # 从增强输入中提取变换信息
        return augmented_inputs, tfms  # 返回增强输入和变换信息