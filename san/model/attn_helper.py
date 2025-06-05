import warnings  # 导入警告模块
from typing import Optional  # 导入Optional类型

import torch  # 导入PyTorch库
from torch import Tensor  # 从PyTorch导入Tensor类型
from torch.nn import functional as F  # 导入PyTorch的函数式接口
from open_clip.transformer import ResidualAttentionBlock  # 从open_clip导入ResidualAttentionBlock


def cross_attn_with_self_bias(
    self, query, key, value, attn_mask=None, need_weights=False, key_padding_mask=None
):
    # 带自偏置的交叉注意力的包装函数，转发调用给具体实现函数
    return cross_attn_with_self_bias_func(
        query,
        key,
        value,
        self.embed_dim,
        self.num_heads,
        self.in_proj_weight,
        self.in_proj_bias,
        self.bias_k,
        self.bias_v,
        self.add_zero_attn,
        self.dropout,
        self.out_proj.weight,
        self.out_proj.bias,
        training=self.training,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
    )


def cross_attn_with_self_bias_func(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    # 带自偏置的交叉注意力函数实现
    tgt_len, bsz, embed_dim = query.size()  # 获取query的维度
    assert embed_dim == embed_dim_to_check  # 检查嵌入维度是否匹配
    # 允许多头注意力有不同的特征维度
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
    assert head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
    scaling = float(head_dim) ** -0.5  # 计算缩放因子

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (
            key is value or torch.equal(key, value)
        ):
            # 自注意力情况
            raise NotImplementedError("self-attention未实现")

        elif key is value or torch.equal(key, value):
            # 编码器-解码器注意力
            # 这是使用in_proj_weight和in_proj_bias的内联投影函数
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)  # 对query进行线性变换

            if key is None:
                assert value is None
                k = None
                v = None
                q_k = None
                q_v = None
            else:
                # 使用in_proj_weight和in_proj_bias的内联投影函数
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)  # 对key进行线性变换并分割为k和v
                q_k, q_v = F.linear(query, _w, _b).chunk(2, dim=-1)  # 对query进行线性变换并分割为q_k和q_v
        else:
            # 使用in_proj_weight和in_proj_bias的内联投影函数
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)  # 对query进行线性变换

            # 使用in_proj_weight和in_proj_bias的内联投影函数
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)  # 对key进行线性变换
            q_k = F.linear(query, _w, _b)  # 对query进行与key相同的线性变换
            # 使用in_proj_weight和in_proj_bias的内联投影函数
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)  # 对value进行线性变换
            q_v = F.linear(query, _w, _b)  # 对query进行与value相同的线性变换
    else:
        # 使用独立的投影权重
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling  # 对q应用缩放因子

    # 注意力掩码处理
    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "attn_mask只支持float、byte和bool类型，不支持{}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "在nn.MultiheadAttention中使用Byte张量作为attn_mask已弃用。请使用bool张量代替。"
            )
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("2D attn_mask的大小不正确。")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("3D attn_mask的大小不正确。")
        else:
            raise RuntimeError(
                "不支持attn_mask的维度{}".format(attn_mask.dim())
            )
        # attn_mask现在是3D的

    # 将ByteTensor类型的key_padding_mask转换为bool类型
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "在nn.MultiheadAttention中使用Byte张量作为key_padding_mask已弃用。请使用bool张量代替。"
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # 处理偏置
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias不能添加到static key。"
            assert static_v is None, "bias不能添加到static value。"
    else:
        assert bias_k is None
        assert bias_v is None

    # 重塑张量维度以便进行多头注意力计算
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        q_k = q_k.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        q_v = q_v.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    # 处理静态key和value
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)  # 获取key的序列长度

    # 处理key填充掩码
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    # 添加零注意力（如果需要）
    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # 计算注意力权重
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    # 应用注意力掩码
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    # 应用key填充掩码
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )
    # attn_out_weights: [bsz * num_heads, tgt_len, src_len] -> [bsz * num_heads, tgt_len, src_len+1]
    self_weight = (q * q_k).sum(dim=-1, keepdim=True)  # [bsz * num_heads, tgt_len, 1] 计算自偏置权重
    total_attn_output_weights = torch.cat([attn_output_weights, self_weight], dim=-1)  # 将交叉注意力权重和自偏置权重拼接
    total_attn_output_weights = F.softmax(total_attn_output_weights, dim=-1)  # 对注意力权重应用softmax
    total_attn_output_weights = F.dropout(
        total_attn_output_weights, p=dropout_p, training=training
    )  # 应用dropout
    attn_output_weights = total_attn_output_weights[
        :, :, :-1
    ]  # [bsz * num_heads, tgt_len, src_len] 提取交叉注意力权重
    self_weight = total_attn_output_weights[:, :, -1:]  # [bsz * num_heads, tgt_len, 1] 提取自偏置权重

    # 计算输出
    attn_output = torch.bmm(
        attn_output_weights, v
    )  # [bsz * num_heads, tgt_len, head_dim] 计算交叉注意力输出
    attn_output = (
        attn_output + self_weight * q_v
    )  # [bsz * num_heads, tgt_len, head_dim] 添加自偏置项
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # 重塑输出维度
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # 应用输出投影

    # 返回结果
    if need_weights:
        # 计算各头的平均注意力权重
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights  # 返回输出和注意力权重
    else:
        return attn_output, None  # 仅返回输出


def cross_attn_layer(self: ResidualAttentionBlock, x, mem, attn_bias):
    # x: [K,N,C] 查询序列
    # mem: [L,N,C] 记忆序列（键和值）
    # attn_bias: [N*num_head,K,L] 注意力偏置
    # return: [K,N,C] 更新后的查询序列
    q_x = self.ln_1(x)  # 对查询应用层归一化
    k_x = v_x = self.ln_1(mem)  # 对键和值应用层归一化
    x = x + self.ls_1(
        cross_attn_with_self_bias(self.attn, q_x, k_x, v_x, attn_mask=attn_bias)[0]
    )  # 残差连接：原始输入 + 多头注意力输出
    x = x + self.ls_2(self.mlp(self.ln_2(x)))  # 残差连接：上一步结果 + MLP输出
    return x


def downsample2d(src, target_shape, method="nearest"):
    # src: [N,C,H,W] 源特征图
    # target_shape: [H',W'] 目标形状
    # return: [N,C,H',W'] 下采样后的特征图
    if method in ["bicubic", "bilinear", "nearest"]:  # 插值方法
        src = F.interpolate(src, size=target_shape, mode=method, align_corners=False)
    elif method == "avg":  # 平均池化
        src = F.adaptive_avg_pool2d(src, output_size=target_shape)
    elif method == "max":  # 最大池化
        src = F.adaptive_max_pool2d(src, output_size=target_shape)
    return src


def resize_pos_embed2d(
    posemb,
    src_shape,
    tgt_shape,
    num_prefix_tokens=1,
    interpolation="bicubic",
    antialias=False,
):
    """调整位置嵌入从src_shape到tgt_shape。posemb: [N,L,C]"""
    if src_shape == tgt_shape:  # 如果源形状和目标形状相同，直接返回
        return posemb
    if num_prefix_tokens:  # 处理前缀令牌（如CLS令牌）
        posemb_prefix, posemb_grid = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb

    # 重塑位置嵌入为2D网格
    posemb_grid = posemb_grid.permute(0, 2, 1).reshape(
        1, -1, src_shape[0], src_shape[1]
    )

    # 插值调整大小
    posemb_grid = F.interpolate(
        posemb_grid,
        size=tgt_shape,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    # 重塑回原始格式
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, tgt_shape[0] * tgt_shape[1], -1
    )
    # 拼接前缀令牌和调整大小后的网格
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb
