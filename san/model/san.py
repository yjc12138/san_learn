from typing import List  # 导入List类型

import open_clip  # 导入open_clip库
import torch  # 导入PyTorch库
from detectron2.config import configurable  # 从detectron2导入配置工具
from detectron2.modeling import META_ARCH_REGISTRY  # 从detectron2导入元架构注册表
from detectron2.modeling.postprocessing import sem_seg_postprocess  # 导入语义分割后处理
from detectron2.structures import ImageList  # 从detectron2导入ImageList结构
from detectron2.utils.memory import retry_if_cuda_oom  # 从detectron2导入CUDA内存不足重试工具
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口

from .clip_utils import (  # 导入CLIP工具函数
    FeatureExtractor,  # 特征提取器
    LearnableBgOvClassifier,  # 可学习背景开放词汇分类器
    PredefinedOvClassifier,  # 预定义开放词汇分类器
    RecWithAttnbiasHead,  # 带注意力偏置的重构头
    get_predefined_templates,  # 获取预定义模板
)
from .criterion import SetCriterion  # 导入集合准则
from .matcher import HungarianMatcher  # 导入匈牙利匹配器
from .side_adapter import build_side_adapter_network  # 导入侧适配器网络构建工具


@META_ARCH_REGISTRY.register()  # 注册为元架构
class SAN(nn.Module):
    @configurable  # 可配置装饰器
    def __init__(
        self,
        *,
        clip_visual_extractor: nn.Module,  # CLIP视觉特征提取器
        clip_rec_head: nn.Module,  # CLIP重构头
        side_adapter_network: nn.Module,  # 侧适配器网络
        ov_classifier: PredefinedOvClassifier,  # 开放词汇分类器
        criterion: SetCriterion,  # 损失计算准则
        size_divisibility: int,  # 大小可除性
        asymetric_input: bool = True,  # 是否使用不对称输入
        clip_resolution: float = 0.5,  # CLIP分辨率
        pixel_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],  # 像素均值
        pixel_std: List[float] = [0.26862954, 0.26130258, 0.27577711],  # 像素标准差
        sem_seg_postprocess_before_inference: bool = False,  # 是否在推理前进行语义分割后处理
    ):
        super().__init__()  # 调用父类初始化方法
        self.asymetric_input = asymetric_input  # 设置是否使用不对称输入
        self.clip_resolution = clip_resolution  # 设置CLIP分辨率
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference  # 设置是否在推理前进行语义分割后处理
        self.size_divisibility = size_divisibility  # 设置大小可除性
        self.criterion = criterion  # 设置损失计算准则

        self.side_adapter_network = side_adapter_network  # 设置侧适配器网络
        self.clip_visual_extractor = clip_visual_extractor  # 设置CLIP视觉特征提取器
        self.clip_rec_head = clip_rec_head  # 设置CLIP重构头
        self.ov_classifier = ov_classifier  # 设置开放词汇分类器
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )  # 注册像素均值缓冲区
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)  # 注册像素标准差缓冲区

    @classmethod
    def from_config(cls, cfg):
        ## 从maskformer2复制
        # 损失参数
        no_object_weight = cfg.MODEL.SAN.NO_OBJECT_WEIGHT  # 无对象权重
        # 损失权重
        class_weight = cfg.MODEL.SAN.CLASS_WEIGHT  # 类别权重
        dice_weight = cfg.MODEL.SAN.DICE_WEIGHT  # DICE权重
        mask_weight = cfg.MODEL.SAN.MASK_WEIGHT  # 掩码权重

        # 构建准则
        matcher = HungarianMatcher(
            cost_class=class_weight,  # 类别成本
            cost_mask=mask_weight,  # 掩码成本
            cost_dice=dice_weight,  # DICE成本
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,  # 训练点数
        )

        weight_dict = {
            "loss_ce": class_weight,  # 交叉熵损失权重
            "loss_mask": mask_weight,  # 掩码损失权重
            "loss_dice": dice_weight,  # DICE损失权重
        }
        aux_weight_dict = {}  # 辅助权重字典
        for i in range(len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS) - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})  # 更新辅助权重字典
        weight_dict.update(aux_weight_dict)  # 更新权重字典
        losses = ["labels", "masks"]  # 损失列表

        criterion = SetCriterion(
            num_classes=cfg.MODEL.SAN.NUM_CLASSES,  # 类别数
            matcher=matcher,  # 匹配器
            weight_dict=weight_dict,  # 权重字典
            eos_coef=no_object_weight,  # 无对象系数
            losses=losses,  # 损失列表
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,  # 训练点数
            oversample_ratio=cfg.MODEL.SAN.OVERSAMPLE_RATIO,  # 过采样比率
            importance_sample_ratio=cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO,  # 重要性采样比率
        )
        ## 复制结束

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.MODEL.SAN.CLIP_MODEL_NAME,  # CLIP模型名称
            pretrained=cfg.MODEL.SAN.CLIP_PRETRAINED_NAME,  # CLIP预训练名称
        )  # 创建CLIP模型和变换
        ov_classifier = LearnableBgOvClassifier(
            model, templates=get_predefined_templates(cfg.MODEL.SAN.CLIP_TEMPLATE_SET)
        )  # 创建可学习背景开放词汇分类器

        clip_visual_extractor = FeatureExtractor(
            model.visual,  # CLIP视觉模型
            last_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,  # 最后一层索引
            frozen_exclude=cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE,  # 排除冻结
        )  # 创建CLIP视觉特征提取器
        clip_rec_head = RecWithAttnbiasHead(
            model.visual,  # CLIP视觉模型
            first_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,  # 第一层索引
            frozen_exclude=cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE,  # 更深层排除冻结
            cross_attn=cfg.MODEL.SAN.REC_CROSS_ATTN,  # 交叉注意力
            sos_token_format=cfg.MODEL.SAN.SOS_TOKEN_FORMAT,  # SOS令牌格式
            sos_token_num=cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,  # SOS令牌数量
            downsample_method=cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD,  # 下采样方法
        )  # 创建带注意力偏置的重构头

        pixel_mean, pixel_std = (
            preprocess.transforms[-1].mean,  # 像素均值
            preprocess.transforms[-1].std,  # 像素标准差
        )
        pixel_mean = [255.0 * x for x in pixel_mean]  # 缩放像素均值
        pixel_std = [255.0 * x for x in pixel_std]  # 缩放像素标准差

        return {
            "clip_visual_extractor": clip_visual_extractor,  # CLIP视觉特征提取器
            "clip_rec_head": clip_rec_head,  # CLIP重构头
            "side_adapter_network": build_side_adapter_network(
                cfg, clip_visual_extractor.output_shapes
            ),  # 侧适配器网络
            "ov_classifier": ov_classifier,  # 开放词汇分类器
            "criterion": criterion,  # 损失计算准则
            "size_divisibility": cfg.MODEL.SAN.SIZE_DIVISIBILITY,  # 大小可除性
            "asymetric_input": cfg.MODEL.SAN.ASYMETRIC_INPUT,  # 不对称输入
            "clip_resolution": cfg.MODEL.SAN.CLIP_RESOLUTION,  # CLIP分辨率
            "sem_seg_postprocess_before_inference": cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,  # 是否在推理前进行语义分割后处理
            "pixel_mean": pixel_mean,  # 像素均值
            "pixel_std": pixel_std,  # 像素标准差
        }

    def forward(self, batched_inputs):
        # 获取每个数据集的分类器权重
        # !! 可以计算一次并保存。它每个数据集只运行一次。
        if "vocabulary" in batched_inputs[0]:  # 如果输入中有词汇表
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(
                    batched_inputs[0]["vocabulary"]
                )
            )  # 通过词汇表获取分类器权重
        else:
            dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]  # 获取数据集名称
            assert (
                len(list(set(dataset_names))) == 1
            ), "一个批次中的所有图像必须来自同一个数据集。"
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
            )  # C+1,ndim  # 通过数据集名称获取分类器权重
        images = [x["image"].to(self.device) for x in batched_inputs]  # 获取图像并移动到设备
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]  # 归一化图像
        images = ImageList.from_tensors(images, self.size_divisibility)  # 创建图像列表
        clip_input = images.tensor  # 获取CLIP输入张量
        if self.asymetric_input:  # 如果使用不对称输入
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear"
            )  # 调整CLIP输入大小
        clip_image_features = self.clip_visual_extractor(clip_input)  # 提取CLIP图像特征
        mask_preds, attn_biases = self.side_adapter_network(
            images.tensor, clip_image_features
        )  # 通过侧适配器网络处理
        # !! 可以优化为并行运行。
        mask_embs = [
            self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,N,C]  # 计算掩码嵌入
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in mask_embs
        ]  # 计算掩码logits
        if self.training:  # 如果是训练模式
            if "instances" in batched_inputs[0]:  # 如果输入中有实例
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # 获取gt实例
                targets = self.prepare_targets(gt_instances, images)  # 准备目标
            else:
                targets = None  # 没有目标
            outputs = {
                "pred_logits": mask_logits[-1],  # 预测logits
                "pred_masks": mask_preds[-1],  # 预测掩码
                "aux_outputs": [
                    {
                        "pred_logits": aux_pred_logits,  # 辅助预测logits
                        "pred_masks": aux_pred_masks,  # 辅助预测掩码
                    }
                    for aux_pred_logits, aux_pred_masks in zip(
                        mask_logits[:-1], mask_preds[:-1]
                    )
                ],  # 辅助输出
            }
            # 基于双向匹配的损失
            losses = self.criterion(outputs, targets)  # 计算损失

            for k in list(losses.keys()):  # 遍历损失键
                if k in self.criterion.weight_dict:  # 如果键在权重字典中
                    losses[k] *= self.criterion.weight_dict[k]  # 应用权重
                else:
                    # 如果没有在`weight_dict`中指定，则移除该损失
                    losses.pop(k)  # 移除损失
            return losses  # 返回损失
        else:  # 如果是推理模式
            mask_preds = mask_preds[-1]  # 获取最后一个预测掩码
            mask_logits = mask_logits[-1]  # 获取最后一个预测logits
            # torch.cuda.empty_cache()
            # 推理
            mask_preds = F.interpolate(
                mask_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )  # 上采样预测掩码
            processed_results = []  # 处理结果列表
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_logits, mask_preds, batched_inputs, images.image_sizes
            ):  # 遍历批次
                height = input_per_image.get("height", image_size[0])  # 获取高度
                width = input_per_image.get("width", image_size[1])  # 获取宽度
                processed_results.append({})  # 添加空字典

                if self.sem_seg_postprocess_before_inference:  # 如果在推理前进行语义分割后处理
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )  # 对预测掩码进行后处理
                    mask_cls_result = mask_cls_result.to(mask_pred_result)  # 移动到相同设备
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )  # 进行语义推理
                if not self.sem_seg_postprocess_before_inference:  # 如果不在推理前进行语义分割后处理
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )  # 对结果进行后处理
                processed_results[-1]["sem_seg"] = r  # 保存语义分割结果
            return processed_results  # 返回处理结果

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]  # 获取填充后的高度和宽度
        new_targets = []  # 新目标列表
        for targets_per_image in targets:  # 遍历每个图像的目标
            # 填充gt
            gt_masks = targets_per_image.gt_masks  # 获取gt掩码
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )  # 创建填充掩码
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks  # 填充gt掩码
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,  # gt类别
                    "masks": padded_masks,  # 填充掩码
                }
            )  # 添加新目标
        return new_targets  # 返回新目标

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # 计算类别softmax，忽略最后一个类别
        mask_pred = mask_pred.sigmoid()  # 对掩码应用sigmoid
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)  # 计算语义分割结果
        return semseg  # 返回语义分割结果

    @property
    def device(self):
        return self.pixel_mean.device  # 返回设备