from typing import List  # 导入List类型
from torch.nn import functional as F  # 导入PyTorch函数式接口
import torch  # 导入PyTorch库
from detectron2.utils.registry import Registry  # 从detectron2导入注册表
from open_clip.model import CLIP  # 从open_clip导入CLIP模型
from torch import nn  # 从PyTorch导入神经网络模块
from .utils import get_labelset_from_dataset  # 从工具导入获取数据集标签集函数
from open_clip import tokenizer  # 从open_clip导入分词器


class PredefinedOvClassifier(nn.Module):
    def __init__(
        self,
        clip_model: CLIP,  # CLIP模型
        cache_feature: bool = True,  # 是否缓存特征
        templates: List[str] = ["a photo of {}"],  # 模板列表
    ):
        # 将CLIP模型复制到此模块
        super().__init__()  # 调用父类初始化方法
        for name, child in clip_model.named_children():  # 遍历命名子模块
            if "visual" not in name:  # 如果不是视觉模块
                self.add_module(name, child)  # 添加模块
        for name, param in clip_model.named_parameters(recurse=False):  # 遍历命名参数
            self.register_parameter(name, param)  # 注册参数
        for name, buffer in clip_model.named_buffers(recurse=False):  # 遍历命名缓冲区
            self.register_buffer(name, buffer)  # 注册缓冲区
        self.templates = templates  # 设置模板
        self._freeze()  # 冻结参数

        self.cache_feature = cache_feature  # 设置是否缓存特征
        if self.cache_feature:  # 如果缓存特征
            self.cache = {}  # 创建缓存字典

    def forward(self, category_names: List[str]):
        text_embed_bucket = []  # 创建文本嵌入桶
        for template in self.templates:  # 遍历模板
            noun_tokens = tokenizer.tokenize(
                [template.format(noun) for noun in category_names]  # 格式化模板
            )  # 分词
            text_inputs = noun_tokens.to(self.text_projection.data.device)  # 移动到设备
            text_embed = self.encode_text(text_inputs, normalize=True)  # 编码文本
            text_embed_bucket.append(text_embed)  # 添加文本嵌入
        text_embed = torch.stack(text_embed_bucket).mean(dim=0)  # 计算平均嵌入
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)  # 归一化
        return text_embed  # 返回文本嵌入

    @torch.no_grad()
    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()  # 获取转换数据类型

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]  # 应用令牌嵌入

        x = x + self.positional_embedding.to(cast_dtype)  # 添加位置嵌入
        x = x.permute(1, 0, 2)  # NLD -> LND  # 调整维度顺序
        x = self.transformer(x, attn_mask=self.attn_mask)  # 应用Transformer
        x = x.permute(1, 0, 2)  # LND -> NLD  # 调整维度顺序
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]  # 应用最终层归一化
        # 从EOT嵌入中提取特征（每个序列中EOT令牌是最高数字）
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # 计算最终嵌入
        return F.normalize(x, dim=-1) if normalize else x  # 返回嵌入

    def get_classifier_by_vocabulary(self, vocabulary: List[str]):
        if self.cache_feature:  # 如果缓存特征
            new_words = [word for word in vocabulary if word not in self.cache]  # 获取新词
            if len(new_words) > 0:  # 如果有新词
                cat_embeddings = self(new_words)  # 计算类别嵌入
                self.cache.update(dict(zip(new_words, cat_embeddings)))  # 更新缓存
            cat_embeddings = torch.stack([self.cache[word] for word in vocabulary])  # 堆叠嵌入
        else:  # 如果不缓存特征
            cat_embeddings = self(vocabulary)  # 计算类别嵌入
        return cat_embeddings  # 返回类别嵌入

    def get_classifier_by_dataset_name(self, dataset_name: str):
        if self.cache_feature:  # 如果缓存特征
            if dataset_name not in self.cache:  # 如果数据集名称不在缓存中
                category_names = get_labelset_from_dataset(dataset_name)  # 获取类别名称
                cat_embeddings = self(category_names)  # 计算类别嵌入
                self.cache[dataset_name] = cat_embeddings  # 缓存嵌入
            cat_embeddings = self.cache[dataset_name]  # 获取缓存嵌入
        else:  # 如果不缓存特征
            category_names = get_labelset_from_dataset(dataset_name)  # 获取类别名称
            cat_embeddings = self(category_names)  # 计算类别嵌入
        return cat_embeddings  # 返回类别嵌入

    def _freeze(self):
        for param in self.parameters():  # 遍历参数
            param.requires_grad = False  # 冻结参数

    def train(self, mode=True):
        super().train(False)  # 始终设置为评估模式


class LearnableBgOvClassifier(PredefinedOvClassifier):
    def __init__(
        self,
        clip_model: CLIP,  # CLIP模型
        cache_feature: bool = True,  # 是否缓存特征
        templates: List[str] = ["a photo of {}"],  # 模板列表
    ):
        super().__init__(clip_model, cache_feature, templates)  # 调用父类初始化方法
        self.bg_embed = nn.Parameter(torch.randn(1, self.text_projection.shape[0]))  # 创建背景嵌入参数
        nn.init.normal_(
            self.bg_embed,
            std=self.bg_embed.shape[1] ** -0.5,
        )  # 初始化背景嵌入

    def get_classifier_by_vocabulary(self, vocabulary: List[str]):
        cat_embedding = super().get_classifier_by_vocabulary(vocabulary)  # 获取类别嵌入
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)  # 添加背景嵌入
        cat_embedding = F.normalize(cat_embedding, p=2, dim=-1)  # 归一化
        return cat_embedding  # 返回类别嵌入

    def get_classifier_by_dataset_name(self, dataset_name: str):
        cat_embedding = super().get_classifier_by_dataset_name(dataset_name)  # 获取类别嵌入
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)  # 添加背景嵌入
        cat_embedding = F.normalize(cat_embedding, p=2, dim=-1)  # 归一化
        return cat_embedding  # 返回类别嵌入