import fvcore.nn.weight_init as weight_init  # 导入权重初始化工具
import torch  # 导入PyTorch库
from detectron2.layers import CNNBlockBase, Conv2d  # 从detectron2导入CNN基础块和二维卷积
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口


class LayerNorm(nn.Module):
    """
    LayerNorm的一个变种，在Transformers中流行，对形状为
    (batch_size, channels, height, width)的输入执行逐点均值和
    方差归一化。
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()  # 调用父类初始化方法
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 创建缩放参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 创建偏置参数
        self.eps = eps  # 设置epsilon值
        self.normalized_shape = (normalized_shape,)  # 设置归一化形状

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)  # 计算通道维度的均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算通道维度的方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用缩放和偏置
        return x


class MLP(nn.Module):
    """非常简单的多层感知器（也称为FFN）"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()  # 调用父类初始化方法
        self.num_layers = num_layers  # 层数
        h = [hidden_dim] * (num_layers - 1)  # 隐藏层维度列表
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )  # 创建层列表

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  # 应用ReLU激活，除了最后一层
        return x


class AddFusion(CNNBlockBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)  # 调用父类初始化方法
        self.input_proj = nn.Sequential(
            LayerNorm(in_channels),  # 层归一化
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,  # 1x1卷积
            ),
        )  # 输入投影
        weight_init.c2_xavier_fill(self.input_proj[-1])  # 使用Xavier初始化卷积层权重

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,L,C] y: [N,C,H,W]
        y = (
            F.interpolate(
                self.input_proj(y.contiguous()),  # 应用输入投影
                size=spatial_shape,  # 调整到目标空间形状
                mode="bilinear",  # 双线性插值
                align_corners=False,
            )  # 上采样
            .permute(0, 2, 3, 1)  # 调整维度顺序
            .reshape(x.shape)  # 重塑为与x相同的形状
        )
        x = x + y  # 元素级加法融合
        return x


def build_fusion_layer(fusion_type: str, in_channels: int, out_channels: int):
    if fusion_type == "add":  # 如果融合类型为"add"
        return AddFusion(in_channels, out_channels)  # 返回加法融合层
    else:
        raise ValueError("未知的融合类型: {}".format(fusion_type))  # 抛出错误