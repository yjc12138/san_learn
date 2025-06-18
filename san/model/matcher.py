# 由Bowen Cheng从https://github.com/facebookresearch/detr/blob/master/models/matcher.py修改
# 从maskformer2复制
"""
用于计算匹配成本并解决相应线性和分配问题的模块。
"""
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch函数式接口
from scipy.optimize import linear_sum_assignment  # 导入线性和分配求解器
from torch import nn  # 从PyTorch导入神经网络模块
from torch.amp import autocast  # 导入自动混合精度

from detectron2.projects.point_rend.point_features import point_sample  # 导入点采样函数


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    计算DICE损失，类似于掩码的广义IoU
    参数:
        inputs: 任意形状的浮点张量。每个样本的预测值。
        targets: 与inputs形状相同的浮点张量。存储inputs中每个元素的二值分类标签
                (0表示负类，1表示正类)。
    """
    inputs = inputs.sigmoid()  # 应用sigmoid函数
    inputs = inputs.flatten(1)  # 展平输入
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)  # 使用爱因斯坦求和计算分子
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]  # 计算分母
    loss = 1 - (numerator + 1) / (denominator + 1)  # 计算损失
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # 使用torch.jit.script编译批量dice损失函数


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    参数:
        inputs: 任意形状的浮点张量。每个样本的预测值。
        targets: 与inputs形状相同的浮点张量。存储inputs中每个元素的二值分类标签
                (0表示负类，1表示正类)。
    返回:
        损失张量
    """
    hw = inputs.shape[1]  # 获取输入的高*宽

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )  # 计算正样本的二值交叉熵
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )  # 计算负样本的二值交叉熵

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )  # 使用爱因斯坦求和计算加权损失

    return loss / hw  # 返回平均损失


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # 使用torch.jit.script编译批量sigmoid交叉熵损失函数


class HungarianMatcher(nn.Module):
    """此类计算目标和网络预测之间的分配
    出于效率考虑，目标不包括no_object。因此，通常情况下，
    预测比目标多。在这种情况下，我们进行一对一匹配，选择最佳预测，
    其余的未匹配（因此被视为非对象）。
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
    ):
        """创建匹配器
        参数:
            cost_class: 这是匹配成本中分类错误的相对权重
            cost_mask: 这是匹配成本中二值掩码的焦点损失的相对权重
            cost_dice: 这是匹配成本中二值掩码的dice损失的相对权重
        """
        super().__init__()  # 调用父类初始化方法
        self.cost_class = cost_class  # 类别成本
        self.cost_mask = cost_mask  # 掩码成本
        self.cost_dice = cost_dice  # dice成本

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "所有成本不能都为0"  # 确保至少有一个成本不为0

        self.num_points = num_points  # 点数量

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """更节省内存的匹配"""
        bs, num_queries = outputs["pred_logits"].shape[:2]  # 获取批量大小和查询数量

        indices = []  # 初始化索引列表

        # 遍历批量
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(
                -1
            )  # [num_queries, num_classes] 计算softmax概率
            tgt_ids = targets[b]["labels"]  # 获取目标ID

            # 计算分类成本。与损失不同，我们不使用NLL，
            # 而是用1 - proba[target class]近似。
            # 1是一个不改变匹配的常数，可以省略。
            cost_class = -out_prob[:, tgt_ids]  # 计算类别成本

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred] 获取预测掩码
            # 在准备目标时，gt掩码已经被填充
            tgt_mask = targets[b]["masks"].to(out_mask)  # 获取目标掩码并移到相同设备

            out_mask = out_mask[:, None]  # 添加维度
            tgt_mask = tgt_mask[:, None]  # 添加维度
            # 所有掩码共享相同的点集，以实现高效匹配!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)  # 随机生成点坐标
            # 获取gt标签
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),  # 重复点坐标
                align_corners=False,  # 不对齐角点
            ).squeeze(1)  # 采样目标掩码

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),  # 重复点坐标
                align_corners=False,  # 不对齐角点
            ).squeeze(1)  # 采样输出掩码

            with autocast("cuda", enabled=False):  # 禁用自动混合精度
                out_mask = out_mask.float()  # 转换为浮点型
                tgt_mask = tgt_mask.float()  # 转换为浮点型
                # 计算掩码之间的焦点损失
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)  # 计算掩码成本

                # 计算掩码之间的dice损失
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)  # 计算dice成本

            # 最终成本矩阵
            C = (
                self.cost_mask * cost_mask  # 掩码成本
                + self.cost_class * cost_class  # 类别成本
                + self.cost_dice * cost_dice  # dice成本
            )
            C = C.reshape(num_queries, -1).cpu()  # 重塑成本矩阵并移到CPU

            indices.append(linear_sum_assignment(C))  # 求解线性和分配问题

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # 转换为张量
                torch.as_tensor(j, dtype=torch.int64),  # 转换为张量
            )
            for i, j in indices  # 遍历索引
        ]  # 返回索引列表

    @torch.no_grad()
    def forward(self, outputs, targets):
        """执行匹配
        参数:
            outputs: 这是一个字典，至少包含以下条目:
                 "pred_logits": 维度为[batch_size, num_queries, num_classes]的张量，包含分类logits
                 "pred_masks": 维度为[batch_size, num_queries, H_pred, W_pred]的张量，包含预测掩码
            targets: 这是一个目标列表（len(targets) = batch_size），其中每个目标是一个包含以下内容的字典:
                 "labels": 维度为[num_target_boxes]的张量（其中num_target_boxes是目标中真实对象的数量），包含类别标签
                 "masks": 维度为[num_target_boxes, H_gt, W_gt]的张量，包含目标掩码
        返回:
            大小为batch_size的列表，包含(index_i, index_j)元组，其中:
                - index_i是选择的预测的索引（按顺序）
                - index_j是相应选择的目标的索引（按顺序）
            对于每个批次元素，有:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)  # 使用内存高效的前向传播

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__  # 设置头部字符串
        body = [
            "cost_class: {}".format(self.cost_class),  # 类别成本
            "cost_mask: {}".format(self.cost_mask),  # 掩码成本
            "cost_dice: {}".format(self.cost_dice),  # dice成本
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]  # 组合行
        return "\n".join(lines)  # 返回字符串表示