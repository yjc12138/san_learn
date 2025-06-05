import logging  # 导入日志模块
from functools import partial  # 导入partial函数，用于创建偏函数
from typing import Dict, List, Tuple  # 导入类型提示

import torch  # 导入PyTorch库
from detectron2.config import configurable  # 从detectron2导入配置工具
from detectron2.layers import ShapeSpec  # 从detectron2导入形状规范
from detectron2.utils.logger import log_first_n  # 从detectron2导入日志工具
from detectron2.utils.registry import Registry  # 从detectron2导入注册表
from timm import create_model  # 从timm导入创建模型函数
from timm.models.vision_transformer import VisionTransformer  # 从timm导入视觉Transformer
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口

from ..layers import MLP, build_fusion_layer  # 从上级模块导入MLP和融合层构建工具
from .timm_wrapper import PatchEmbed  # 导入补丁嵌入包装器

SIDE_ADAPTER_REGISTRY = Registry("SIDE_ADAPTER")  # 创建侧适配器注册表
SIDE_ADAPTER_REGISTRY.__doc__ = """
侧适配器的注册表。
"""


def build_side_adapter_network(cfg, input_shape):
    name = cfg.MODEL.SIDE_ADAPTER.NAME  # 获取侧适配器名称
    return SIDE_ADAPTER_REGISTRY.get(name)(cfg, input_shape)  # 创建并返回侧适配器网络


class MLPMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,  # 输入通道数
        total_heads: int = 1,  # 总头数
        total_layers: int = 1,  # 总层数
        embed_channels: int = 256,  # 嵌入通道数
        mlp_channels: int = 256,  # MLP通道数
        mlp_num_layers: int = 3,  # MLP层数
        rescale_attn_bias: bool = False,  # 是否重新缩放注意力偏置
    ):
        super().__init__()  # 调用父类初始化方法
        self.total_heads = total_heads  # 设置总头数
        self.total_layers = total_layers  # 设置总层数

        dense_affine_func = partial(nn.Conv2d, kernel_size=1)  # 创建1x1卷积的偏函数
        # 查询分支
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels, mlp_num_layers)  # 创建查询MLP
        # 像素分支
        self.pix_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )  # 创建像素MLP
        # 注意力偏置分支
        self.attn_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels * self.total_heads * self.total_layers,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )  # 创建注意力MLP
        if rescale_attn_bias:  # 如果重新缩放注意力偏置
            self.bias_scaling = nn.Linear(1, 1)  # 创建线性缩放层
        else:
            self.bias_scaling = nn.Identity()  # 创建恒等映射

    def forward(
        self, query: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # query: [B,N,C] 查询张量
        # x: [B,C,H,W] 特征图张量
        query = self.query_mlp(query)  # 应用查询MLP
        pix = self.pix_mlp(x)  # 应用像素MLP
        b, c, h, w = pix.shape  # 获取形状
        # 预测掩码
        mask_preds = torch.einsum("bqc,bchw->bqhw", query, pix)  # 计算掩码预测
        # 生成注意力偏置
        attn = self.attn_mlp(x)  # 应用注意力MLP
        attn = attn.reshape(b, self.total_layers, self.total_heads, c, h, w)  # 重塑形状
        attn_bias = torch.einsum("bqc,blnchw->blnqhw", query, attn)  # 计算注意力偏置
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)  # 应用偏置缩放
        attn_bias = attn_bias.chunk(self.total_layers, dim=1)  # 按层分块
        attn_bias = [attn.squeeze(1) for attn in attn_bias]  # 压缩维度
        return mask_preds, attn_bias  # 返回掩码预测和注意力偏置


@SIDE_ADAPTER_REGISTRY.register()  # 注册为侧适配器
class RegionwiseSideAdapterNetwork(nn.Module):
    @configurable  # 可配置装饰器
    def __init__(
        self,
        vit_model: VisionTransformer,  # ViT模型
        fusion_layers: nn.ModuleList,  # 融合层列表
        mask_decoder: nn.Module,  # 掩码解码器
        num_queries: int,  # 查询数量
        fusion_map: Dict[int, int],  # 融合映射
        deep_supervision_idxs: List[int],  # 深度监督索引
    ):
        super().__init__()  # 调用父类初始化方法
        # 移除cls令牌
        if vit_model.cls_token is not None:
            vit_model.pos_embed = nn.Parameter(vit_model.pos_embed[:, 1:, ...])  # 更新位置嵌入
        del vit_model.cls_token  # 删除cls令牌
        vit_model.cls_token = None  # 设置为None
        # 删除输出归一化
        del vit_model.norm  # 删除归一化层
        vit_model.norm = nn.Identity()  # 设置为恒等映射
        self.vit_model = vit_model  # 设置ViT模型

        self.num_queries = num_queries  # 设置查询数量
        self.num_features = vit_model.num_features  # 设置特征数量
        # 添加查询令牌
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, self.num_features))  # 创建查询嵌入参数
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, self.num_features)
        )  # 创建查询位置嵌入参数
        nn.init.normal_(self.query_embed, std=0.02)  # 初始化查询嵌入
        nn.init.normal_(self.query_pos_embed, std=0.02)  # 初始化查询位置嵌入
        self.fusion_layers = fusion_layers  # 设置融合层
        self.fusion_map = fusion_map  # 设置融合映射
        self.mask_decoder = mask_decoder  # 设置掩码解码器
        # 用于训练
        self.deep_supervision_idxs = deep_supervision_idxs  # 设置深度监督索引

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        vit = create_model(
            cfg.MODEL.SIDE_ADAPTER.VIT_NAME,  # ViT名称
            cfg.MODEL.SIDE_ADAPTER.PRETRAINED,  # 预训练
            img_size=cfg.MODEL.SIDE_ADAPTER.IMAGE_SIZE,  # 图像大小
            drop_path_rate=cfg.MODEL.SIDE_ADAPTER.DROP_PATH_RATE,  # 路径丢弃率
            fc_norm=False,  # 不使用FC归一化
            num_classes=0,  # 类别数为0
            embed_layer=PatchEmbed,  # 嵌入层
        )  # 创建ViT模型
        # ["0->0","3->1","6->2","9->3"]
        fusion_map: List[str] = cfg.MODEL.SIDE_ADAPTER.FUSION_MAP  # 获取融合映射

        x2side_map = {int(j): int(i) for i, j in [x.split("->") for x in fusion_map]}  # 创建映射字典
        # 构建融合层
        fusion_type: str = cfg.MODEL.SIDE_ADAPTER.FUSION_TYPE  # 获取融合类型
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer(
                    fusion_type, input_shape[src_idx].channels, vit.num_features
                )
                for tgt_idx, src_idx in x2side_map.items()  # 遍历映射
            }
        )  # 创建融合层字典
        # 构建掩码解码器
        return {
            "vit_model": vit,  # ViT模型
            "num_queries": cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,  # 查询数量
            "fusion_layers": fusion_layers,  # 融合层
            "fusion_map": x2side_map,  # 融合映射
            "mask_decoder": MLPMaskDecoder(
                in_channels=vit.num_features,  # 输入通道数
                total_heads=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_HEADS,  # 总头数
                total_layers=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_LAYERS,  # 总层数
                embed_channels=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.EMBED_CHANNELS,  # 嵌入通道数
                mlp_channels=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_CHANNELS,  # MLP通道数
                mlp_num_layers=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_NUM_LAYERS,  # MLP层数
                rescale_attn_bias=cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.RESCALE_ATTN_BIAS,  # 是否重新缩放注意力偏置
            ),  # 创建MLP掩码解码器
            "deep_supervision_idxs": cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS,  # 深度监督索引
        }

    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        features = self.forward_features(image, clip_features)  # 前向计算特征
        return self.decode_masks(features)  # 解码掩码

    def decode_masks(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        if not self.training:  # 如果不是训练模式
            features = [features[-1]]  # 只使用最后一个特征
        mask_preds = []  # 掩码预测列表
        attn_biases = []  # 注意力偏置列表
        for feature in features:  # 遍历特征
            mask_pred, attn_bias = self.mask_decoder(**feature)  # 解码掩码和注意力偏置
            mask_preds.append(mask_pred)  # 添加掩码预测
            attn_biases.append(attn_bias)  # 添加注意力偏置
        return mask_preds, attn_biases  # 返回掩码预测和注意力偏置

    def forward_features(
        self, image: torch.Tensor, clip_features: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        x, (h, w) = self.vit_model.patch_embed(image)  # 计算补丁嵌入
        L = x.shape[1]  # 令牌长度
        pos_embed = self.vit_model.pos_embed  # 获取位置嵌入
        ori_h, ori_w = self.vit_model.patch_embed.grid_size  # 获取原始网格大小
        if pos_embed.shape[1] != L:  # 如果位置嵌入形状不匹配
            pos_embed = (
                F.interpolate(
                    pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                    size=[h, w],
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )  # 插值调整位置嵌入大小
        pos_embed = torch.cat(
            [self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1
        )  # 拼接位置嵌入
        x = torch.cat(
            [self.query_embed.expand(x.shape[0], -1, -1), x],
            dim=1,
        )  # B, Q+L, C  # 拼接查询嵌入和特征
        x = x + pos_embed  # 添加位置嵌入
        x = self.vit_model.norm_pre(x)  # 应用预归一化
        x = self.fuse(0, x, clip_features, (h, w))  # 融合特征
        outs = []  # 输出列表
        for i, blk in enumerate(self.vit_model.blocks, start=1):  # 遍历ViT块
            x = blk(x)  # 应用ViT块
            x = self.fuse(i, x, clip_features, (h, w))  # 融合特征
            if i in self.deep_supervision_idxs:  # 如果是深度监督索引
                outs.append(
                    {
                        "query": x[:, :-L, ...],  # 查询特征
                        "x": x[:, -L:, ...]
                        .permute(0, 2, 1)
                        .reshape(x.shape[0], x.shape[-1], h, w),  # 重塑特征
                    }
                )  # 添加输出

            if i < len(self.vit_model.blocks):  # 如果不是最后一个块
                x = x + pos_embed  # 添加位置嵌入

        return outs  # 返回输出列表

    def fuse(
        self,
        block_idx: int,
        x: torch.Tensor,
        clip_features: List[torch.Tensor],
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if block_idx in self.fusion_map:  # 如果块索引在融合映射中
            src_idx = self.fusion_map[block_idx]  # 获取源索引
            L = spatial_shape[0] * spatial_shape[1]  # 计算空间大小
            x = torch.cat(
                [
                    x[:, :-L, ...],  # 查询特征
                    self.fusion_layers[f"layer_{block_idx}"](
                        x[:, -L:, ...], clip_features[src_idx], spatial_shape
                    ),  # 应用融合层
                ],
                dim=1,
            )  # 拼接特征
            log_first_n(
                logging.INFO,
                f"融合CLIP {src_idx} 到 {block_idx}",
                len(self.fusion_map),
            )  # 记录日志
        return x  # 返回融合后的特征