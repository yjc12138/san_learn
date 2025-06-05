import warnings  # 导入警告模块
from torch import nn  # 从PyTorch导入神经网络模块
from timm.models.vision_transformer import _create_vision_transformer  # 从timm导入创建视觉Transformer函数
from timm.models import register_model  # 从timm导入模型注册装饰器
from timm.models.layers import to_2tuple  # 从timm导入转换为2元组函数


class PatchEmbed(nn.Module):
    """2D图像到补丁嵌入。修改原始实现以允许返回2D补丁大小。"""

    def __init__(
        self,
        img_size=224,  # 图像大小
        patch_size=16,  # 补丁大小
        in_chans=3,  # 输入通道数
        embed_dim=768,  # 嵌入维度
        norm_layer=None,  # 归一化层
        flatten=True,  # 是否展平
        bias=True,  # 是否使用偏置
        **kwargs  # 额外参数
    ):
        super().__init__()  # 调用父类初始化方法
        if len(kwargs)>0:
            warnings.warn(f"未使用的kwargs:{kwargs}。")  # 警告未使用的参数
        img_size = to_2tuple(img_size)  # 转换图像大小为2元组
        patch_size = to_2tuple(patch_size)  # 转换补丁大小为2元组
        self.img_size = img_size  # 设置图像大小
        self.patch_size = patch_size  # 设置补丁大小
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 计算网格大小
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 计算补丁数量
        self.flatten = flatten  # 设置是否展平

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )  # 创建投影卷积层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 创建归一化层

    def forward(self, x):
        x = self.proj(x)  # 应用投影
        h, w = x.shape[-2:]  # 获取高度和宽度
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  # 展平并转置
        x = self.norm(x)  # 应用归一化
        return x, (h, w)  # 返回结果和形状


@register_model
def vit_w144n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained  # 确保不使用预训练
    model_kwargs = dict(patch_size=16, embed_dim=144, depth=8, num_heads=6, **kwargs)  # 设置模型参数
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )  # 创建视觉Transformer
    return model  # 返回模型


@register_model
def vit_w192n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained  # 确保不使用预训练
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=8, num_heads=6, **kwargs)  # 设置模型参数
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )  # 创建视觉Transformer
    return model  # 返回模型


@register_model
def vit_w240n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained  # 确保不使用预训练
    model_kwargs = dict(patch_size=16, embed_dim=240, depth=8, num_heads=6, **kwargs)  # 设置模型参数
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )  # 创建视觉Transformer
    return model  # 返回模型


@register_model
def vit_w288n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained  # 确保不使用预训练
    model_kwargs = dict(patch_size=16, embed_dim=288, depth=8, num_heads=6, **kwargs)  # 设置模型参数
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )  # 创建视觉Transformer
    return model  # 返回模型