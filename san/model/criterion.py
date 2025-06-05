# 从maskformer2复制，作者Bowen Cheng
import logging  # 导入日志模块

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import nn  # 从PyTorch导入神经网络模块

from detectron2.utils.comm import get_world_size  # 从detectron2导入获取世界大小的函数
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,  # 导入获取不确定点坐标的函数
    point_sample,  # 导入点采样函数
)

from san.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list  # 导入辅助函数


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    计算DICE损失，类似于掩码的广义IoU
    参数:
        inputs: 任意形状的浮点张量。每个样本的预测值。
        targets: 与inputs形状相同的浮点张量。存储inputs中每个元素的二值分类标签
                (0表示负类，1表示正类)。
    """
    inputs = inputs.sigmoid()  # 对输入应用sigmoid函数
    inputs = inputs.flatten(1)  # 将输入展平
    numerator = 2 * (inputs * targets).sum(-1)  # 计算分子：2*TP
    denominator = inputs.sum(-1) + targets.sum(-1)  # 计算分母：预测正类数+实际正类数
    loss = 1 - (numerator + 1) / (denominator + 1)  # 计算损失：1-DICE系数
    return loss.sum() / num_masks  # 返回平均损失


dice_loss_jit = torch.jit.script(dice_loss)  # 使用torch.jit.script编译dice_loss函数


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    参数:
        inputs: 任意形状的浮点张量。每个样本的预测值。
        targets: 与inputs形状相同的浮点张量。存储inputs中每个元素的二值分类标签
                (0表示负类，1表示正类)。
    返回:
        损失张量
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # 计算二值交叉熵损失

    return loss.mean(1).sum() / num_masks  # 返回平均损失


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # 使用torch.jit.script编译sigmoid_ce_loss函数


def calculate_uncertainty(logits):
    """
    我们将不确定性估计为类别`classes`中前景类的logit预测与0.0之间的L1距离。
    参数:
        logits (Tensor): 形状为(R, 1, ...)的张量，用于类别特定或类别无关，
            其中R是所有图像中预测掩码的总数，C是前景类的数量。值为logits。
    返回:
        scores (Tensor): 形状为(R, 1, ...)的张量，包含不确定性分数，
            最不确定的位置具有最高的不确定性分数。
    """
    assert logits.shape[1] == 1  # 确保logits的第二个维度为1
    gt_class_logits = logits.clone()  # 克隆logits
    return -(torch.abs(gt_class_logits))  # 返回logits绝对值的负数作为不确定性


class SetCriterion(nn.Module):
    """此类计算DETR的损失。
    过程分两步进行：
        1) 我们计算真实框和模型输出之间的匈牙利分配
        2) 我们监督每对匹配的真实值/预测值（监督类别和框）
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        """
        创建准则。
        参数:
            num_classes: 对象类别的数量，省略特殊的无对象类别
            matcher: 能够计算目标和建议之间匹配的模块
            weight_dict: 字典，包含损失名称作为键，相对权重作为值
            eos_coef: 应用于无对象类别的相对分类权重
            losses: 要应用的所有损失列表。查看get_loss函数了解可用损失列表。
        """
        super().__init__()  # 调用父类初始化方法
        self.num_classes = num_classes  # 类别数量
        self.matcher = matcher  # 匹配器
        self.weight_dict = weight_dict  # 权重字典
        self.eos_coef = eos_coef  # 无对象类的系数
        self.losses = losses  # 损失列表
        empty_weight = torch.ones(self.num_classes + 1)  # 创建权重张量
        empty_weight[-1] = self.eos_coef  # 设置无对象类的权重
        self.register_buffer("empty_weight", empty_weight)  # 注册非参数buffer

        # 点式掩码损失参数
        self.num_points = num_points  # 点数量
        self.oversample_ratio = oversample_ratio  # 过采样比率
        self.importance_sample_ratio = importance_sample_ratio  # 重要性采样比率

    def loss_labels(self, outputs, targets, indices, num_masks):
        """分类损失（NLL）
        targets字典必须包含键"labels"，包含维度为[nb_target_boxes]的张量
        """
        assert "pred_logits" in outputs  # 确保输出中有预测logits
        src_logits = outputs["pred_logits"].float()  # 获取预测logits

        idx = self._get_src_permutation_idx(indices)  # 获取源排列索引
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )  # 获取目标类别
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )  # 创建目标类别张量
        target_classes[idx] = target_classes_o  # 填充目标类别

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )  # 计算交叉熵损失
        losses = {"loss_ce": loss_ce}  # 创建损失字典
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """计算与掩码相关的损失：焦点损失和dice损失。
        targets字典必须包含键"masks"，包含维度为[nb_target_boxes, h, w]的张量
        """
        assert "pred_masks" in outputs  # 确保输出中有预测掩码

        src_idx = self._get_src_permutation_idx(indices)  # 获取源排列索引
        tgt_idx = self._get_tgt_permutation_idx(indices)  # 获取目标排列索引
        src_masks = outputs["pred_masks"]  # 获取预测掩码
        src_masks = src_masks[src_idx]  # 根据索引选择预测掩码
        masks = [t["masks"] for t in targets]  # 获取目标掩码
        # TODO 使用valid来掩盖由于填充而无效的区域
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()  # 分解嵌套张量
        target_masks = target_masks.to(src_masks)  # 将目标掩码移到相同设备
        target_masks = target_masks[tgt_idx]  # 根据索引选择目标掩码

        # 不需要上采样预测，因为我们使用归一化坐标:)
        # N x 1 x H x W
        src_masks = src_masks[:, None]  # 添加维度
        target_masks = target_masks[:, None]  # 添加维度

        with torch.no_grad():  # 不计算梯度
            # 采样点坐标
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # 获取gt标签
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)  # 采样点logits

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }  # 计算掩码损失和dice损失

        del src_masks  # 删除临时变量
        del target_masks  # 删除临时变量
        return losses

    def _get_src_permutation_idx(self, indices):
        # 根据索引排列预测
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )  # 批次索引
        src_idx = torch.cat([src for (src, _) in indices])  # 源索引
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 根据索引排列目标
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )  # 批次索引
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])  # 目标索引
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,  # 标签损失函数
            "masks": self.loss_masks,  # 掩码损失函数
        }
        assert loss in loss_map, f"你确定要计算{loss}损失吗？"
        return loss_map[loss](outputs, targets, indices, num_masks)  # 返回指定的损失

    def forward(self, outputs, targets):
        """执行损失计算。
        参数:
             outputs: 张量字典，有关格式，请参阅模型的输出规范
             targets: 字典列表，使得len(targets) == batch_size。
                      每个字典中的预期键取决于应用的损失，请参阅每个损失的文档
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}  # 移除辅助输出

        # 检索最后一层的输出与目标之间的匹配
        indices = self.matcher(outputs_without_aux, targets)  # 计算匹配索引

        # 计算所有节点的平均目标框数量，用于归一化
        num_masks = sum(len(t["labels"]) for t in targets)  # 计算掩码总数
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )  # 转换为张量
        if is_dist_avail_and_initialized():  # 如果分布式环境可用
            torch.distributed.all_reduce(num_masks)  # 对掩码数量进行all_reduce操作
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()  # 计算平均掩码数量

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))  # 更新损失字典

        # 如果有辅助损失，我们对每个中间层的输出重复此过程。
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)  # 计算辅助输出的匹配索引
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )  # 计算损失
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}  # 重命名损失键
                    losses.update(l_dict)  # 更新损失字典

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__  # 设置头部字符串
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),  # 匹配器表示
            "losses: {}".format(self.losses),  # 损失列表
            "weight_dict: {}".format(self.weight_dict),  # 权重字典
            "num_classes: {}".format(self.num_classes),  # 类别数量
            "eos_coef: {}".format(self.eos_coef),  # 无对象类系数
            "num_points: {}".format(self.num_points),  # 点数量
            "oversample_ratio: {}".format(self.oversample_ratio),  # 过采样比率
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),  # 重要性采样比率
        ]
        _repr_indent = 4  # 设置缩进
        lines = [head] + [" " * _repr_indent + line for line in body]  # 组合行
        return "\n".join(lines)  # 返回字符串表示