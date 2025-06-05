from typing import List  # 导入List类型
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口
from open_clip.transformer import VisionTransformer  # 从open_clip导入视觉Transformer
from detectron2.layers import ShapeSpec  # 从detectron2导入形状规范
from ..attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d  # 导入注意力辅助函数


class ClipOutput(dict):
    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类初始化方法
        self.spacial_shape = spacial_shape  # 设置空间形状

    def save(self, idx: int, clip_feat: torch.Tensor):
        l, n, c = clip_feat.shape  # 获取特征形状
        self[idx] = (
            clip_feat[1:].permute(1, 2, 0).reshape(n, c, *self.spacial_shape)
        )  # n, c, h, w  # 保存特征
        self[f"{idx}_cls_token"] = clip_feat[0:1]  # 1, n, c  # 保存CLS令牌


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,  # 视觉编码器
        last_layer_idx: int = -1,  # 最后一层索引
        frozen_exclude=[],  # 排除冻结的参数
    ):
        super().__init__()  # 调用父类初始化方法
        self.output_tokens = visual_encoder.output_tokens  # 设置输出令牌
        self.image_size = visual_encoder.image_size  # 设置图像大小
        self.patch_size = visual_encoder.patch_size  # 设置补丁大小
        self.grid_size = visual_encoder.grid_size  # 设置网格大小
        self.num_features = visual_encoder.ln_pre.normalized_shape[0]  # 设置特征数量

        self.input_patchnorm = visual_encoder.input_patchnorm  # 设置输入补丁归一化
        self.patchnorm_pre_ln = visual_encoder.patchnorm_pre_ln  # 设置补丁归一化前的层归一化
        self.conv1 = visual_encoder.conv1  # 设置第一个卷积层

        # 类别嵌入和位置嵌入
        self.class_embedding = visual_encoder.class_embedding  # 设置类别嵌入
        self.positional_embedding = visual_encoder.positional_embedding  # 设置位置嵌入
        # 设置补丁丢弃率为0.表示禁用，这个函数将是恒等函数
        self.patch_dropout = visual_encoder.patch_dropout  # 设置补丁丢弃
        self.ln_pre = visual_encoder.ln_pre  # 设置前置层归一化
        if last_layer_idx == -1:  # 如果最后一层索引为-1
            self.resblocks = visual_encoder.transformer.resblocks  # 使用所有残差块
            self.last_output_idx = len(self.resblocks) + 1  # 设置最后输出索引
        else:
            self.resblocks = visual_encoder.transformer.resblocks[:last_layer_idx]  # 使用部分残差块
            self.last_output_idx = last_layer_idx + 1  # 设置最后输出索引
        #
        self.frozen_exclude = frozen_exclude  # 设置排除冻结的参数
        self._freeze(self.frozen_exclude)  # 冻结参数

    def forward(self, x: torch.Tensor):
        if self.input_patchnorm:  # 如果使用输入补丁归一化
            raise NotImplementedError("input_patchnorm尚未实现。")
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]  # 应用第一个卷积层
            _, _, h, w = x.shape  # 获取形状
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  # 重塑形状
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 调整维度顺序

        # 类别嵌入和位置嵌入
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),  # 添加类别嵌入
                x,  # 添加特征
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]  # 拼接
        pos_embed = self.positional_embedding.to(x.dtype)  # 转换位置嵌入数据类型
        pos_embed = resize_pos_embed2d(pos_embed[None, ...], self.grid_size, (h, w))[0]  # 调整位置嵌入大小
        x = x + pos_embed  # 添加位置嵌入

        # 补丁丢弃率为0.表示禁用，这个函数将不做任何事情，只返回传入的内容
        x = self.patch_dropout(x)  # 应用补丁丢弃
        x = self.ln_pre(x)  # 应用前置层归一化
        x = x.permute(1, 0, 2)  # NLD -> LND  # 调整维度顺序

        outputs = ClipOutput(spacial_shape=(h, w))  # 创建输出对象
        outputs.save(0, x)  # 保存初始特征
        for i, resblock in enumerate(self.resblocks, start=1):  # 遍历残差块
            x = resblock(x)  # 应用残差块
            outputs.save(i, x)  # 保存特征
        return outputs  # 返回输出

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:  # 如果排除所有参数
            return  # 不冻结任何参数
        for name, param in self.named_parameters():  # 遍历命名参数
            if not any([exclude in name for exclude in frozen_exclude]):  # 如果参数名不在排除列表中
                param.requires_grad = False  # 冻结参数

    @property
    def output_shapes(self):
        return {
            i: ShapeSpec(channels=self.num_features)
            for i in range(self.last_output_idx)
        }  # 返回输出形状

    @property
    def size_divisibility(self):
        return self.patch_size[0]  # 返回大小可除性


class RecWithAttnbiasHead(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,  # 视觉编码器
        first_layer_idx: int = 0,  # 第一层索引
        frozen_exclude: List[str] = [],  # 排除冻结的参数
        sos_token_format: str = "cls_token",  # SOS令牌格式
        sos_token_num: int = 1,  # SOS令牌数量
        cross_attn: bool = True,  # 是否使用交叉注意力
        downsample_method: str = "bilinear",  # 下采样方法
    ):
        super().__init__()  # 调用父类初始化方法
        self.output_tokens = visual_encoder.output_tokens  # 设置输出令牌
        self.output_dim = visual_encoder.output_dim  # 设置输出维度
        self.first_layer_idx = first_layer_idx  # 设置第一层索引
        self.cross_attn = cross_attn  # 设置是否使用交叉注意力
        self.downsample_method = downsample_method  # 设置下采样方法

        if first_layer_idx < 0:  # 如果第一层索引小于0
            raise NotImplementedError("first_layer_idx < 0 尚未实现。")  # 抛出错误
        self.resblocks = visual_encoder.transformer.resblocks[first_layer_idx:]  # 使用部分残差块
        self.global_average_pool = visual_encoder.global_average_pool  # 设置全局平均池化
        self.attn_pool = visual_encoder.attn_pool  # 设置注意力池化
        assert (
            self.attn_pool is None
        ), "带有attn_pool的识别尚未实现。"  # 确保注意力池化为None
        assert (
            not self.global_average_pool
        ), "带有global_average_pool的识别尚未实现。"  # 确保全局平均池化为False
        self.ln_post = visual_encoder.ln_post  # 设置后置层归一化
        self.proj = visual_encoder.proj  # 设置投影

        self.sos_token_format = sos_token_format  # 设置SOS令牌格式
        self.sos_token_num = sos_token_num  # 设置SOS令牌数量
        self.frozen_exclude = frozen_exclude  # 设置排除冻结的参数

        if sos_token_format in ["learnable_token", "pos_embedding"]:  # 如果SOS令牌格式为可学习令牌或位置嵌入
            self.sos_token = nn.Parameter(
                torch.randn(sos_token_num, 1, self.proj.shape[0])
            )  # 创建SOS令牌参数
            nn.init.normal_(self.sos_token, std=0.02)  # 初始化SOS令牌
            self.frozen_exclude.append("sos_token")  # 将SOS令牌添加到排除冻结列表
        self._freeze(self.frozen_exclude)  # 冻结参数

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:  # 如果排除所有参数
            return  # 不冻结任何参数
        for name, param in self.named_parameters():  # 遍历命名参数
            if not any([exclude in name for exclude in frozen_exclude]):  # 如果参数名不在排除列表中
                param.requires_grad = False  # 冻结参数

    def forward(self, features, attn_bias, normalize: bool = False):
        # 构建CLIP影子特征
        cls_token = features[f"{self.first_layer_idx}_cls_token"]  # 1,n,c  # 获取CLS令牌
        pix_feat = features[self.first_layer_idx]  # n,c,h,w  # 获取像素特征
        n, c, h, w = pix_feat.shape  # 获取形状
        x = torch.cat(
            [cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)]
        )  # 1+l,n,c  # 拼接特征

        # 构建SOS令牌
        if self.sos_token_format == "cls_token":  # 如果SOS令牌格式为CLS令牌
            sos_token = cls_token.repeat(self.sos_token_num, 1, 1)  # 复制CLS令牌
        elif self.sos_token_format == "learnable_token":  # 如果SOS令牌格式为可学习令牌
            sos_token = self.sos_token.expand(-1, n, -1)  # 扩展SOS令牌
        elif self.sos_token_format == "pos_embedding":  # 如果SOS令牌格式为位置嵌入
            sos_token = self.sos_token.expand(-1, n, -1) + cls_token  # 扩展SOS令牌并添加CLS令牌

        # 构建注意力偏置
        attn_biases = self._build_attn_biases(attn_bias, target_shape=(h, w))  # 构建注意力偏置
        if self.cross_attn:  # 如果使用交叉注意力
            for i, resblock in enumerate(self.resblocks):  # 遍历残差块
                if self.cross_attn:  # 如果使用交叉注意力
                    sos_token = cross_attn_layer(
                        resblock,
                        sos_token,
                        x[1:,],
                        attn_biases[i],
                    )  # 应用交叉注意力层
                    if i < len(self.resblocks) - 1:  # 如果不是最后一个残差块
                        x = resblock(x)  # 应用残差块
        else:  # 如果不使用交叉注意力
            x = torch.cat([sos_token, x], dim=0)  # 拼接特征
            for i, resblock in enumerate(self.resblocks):  # 遍历残差块
                x = resblock(x, attn_mask=attn_biases[i])  # 应用残差块
            sos_token = x[: self.sos_token_num]  # 获取SOS令牌

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD  # 调整维度顺序

        sos_token = self.ln_post(sos_token)  # 应用后置层归一化

        if self.proj is not None:  # 如果存在投影
            sos_token = sos_token @ self.proj  # 应用投影
        if normalize:  # 如果需要归一化
            sos_token = F.normalize(sos_token, dim=-1)  # 归一化
        return sos_token  # 返回SOS令牌

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []  # 创建格式化注意力偏置列表
        for attn_bias in attn_biases:  # 遍历注意力偏置
            # 转换为适当的格式: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape  # 获取形状
            # 重塑并下采样
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                target_shape,
                method=self.downsample_method,
            )  # 下采样注意力偏置
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)  # 重塑形状
            true_num_head = self.resblocks[0].attn.num_heads  # 获取真实头数
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head}不受支持。"  # 确保头数合法
            if num_head == 1:  # 如果头数为1
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)  # 重复注意力偏置
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)  # 重塑形状
            L = attn_bias.shape[-1]  # 获取长度
            if self.cross_attn:  # 如果使用交叉注意力
                # [n*num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)  # 添加注意力偏置
            else:  # 如果不使用交叉注意力
                # [n*num_head, num_sos+1+L, num_sos+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)  # 创建新的注意力偏置
                new_attn_bias[:, :num_sos] = -100  # 设置值
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0  # 设置对角线值
                new_attn_bias[:num_sos, num_sos] = -100  # 设置值
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone()
                )  # 扩展注意力偏置
                new_attn_bias[..., :num_sos, -L:] = attn_bias  # 设置值
                formatted_attn_biases.append(new_attn_bias)  # 添加注意力偏置

        if len(formatted_attn_biases) == 1:  # 如果只有一个注意力偏置
            formatted_attn_biases = [formatted_attn_biases[0] for _ in self.resblocks]  # 复制注意力偏置
        return formatted_attn_biases  # 返回格式化注意力偏置