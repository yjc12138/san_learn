import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry


class SAMFeatureExtractor(nn.Module):
    """
    SAM特征提取器，用于从SAM模型中提取图像特征，并与CLIP特征融合
    """
    def __init__(self, checkpoint_path, model_type="vit_b", frozen=True, exclude_pos=True):
        """
        初始化SAM特征提取器
        
        Args:
            checkpoint_path: SAM模型权重路径
            model_type: SAM模型类型，默认为vit_b
            frozen: 是否冻结SAM模型参数
            exclude_pos: 是否排除位置编码参数的冻结
        """
        super().__init__()
        # 加载SAM模型
        try:
            self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            
            # 提取图像编码器
            self.image_encoder = self.sam_model.image_encoder
            
            # 添加LayerNorm层用于特征归一化
            self.layer_norm = nn.LayerNorm(256)  # SAM输出特征维度为256
            
            # 冻结参数
            if frozen:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                    
                # 如果exclude_pos为True，则解冻位置编码参数
                if exclude_pos and hasattr(self.image_encoder, 'pos_embed'):
                    self.image_encoder.pos_embed.requires_grad = True
                    
            print(f"成功加载SAM模型 ({model_type})")
        except Exception as e:
            print(f"警告: 加载SAM模型时出错: {e}")
            # 创建一个空的图像编码器作为备份
            self.image_encoder = None
            self.layer_norm = nn.LayerNorm(256)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [B, C, H, W]
            
        Returns:
            sam_features: SAM特征 [B, 256, H', W']
        """
        # 如果图像编码器为None，返回零张量
        if self.image_encoder is None:
            # 返回与输入相同batch size的零张量，但通道数为256，空间尺寸为输入的1/16
            b, _, h, w = x.shape
            return torch.zeros(b, 256, h//16, w//16, device=x.device)
        
        # 尝试使用GPU，如果失败则切换到CPU
        try:
            # 通过SAM图像编码器提取特征
            with torch.no_grad() if not self.training else torch.enable_grad():
                sam_features = self.image_encoder(x)
            
            return sam_features
        except RuntimeError as e:
            if "CUDA" in str(e) or "GET was unable to find an engine" in str(e):
                print("警告: CUDA错误，尝试在CPU上运行SAM...")
                # 将模型和输入移动到CPU
                device = x.device
                cpu_encoder = self.image_encoder.cpu()
                cpu_x = x.cpu()
                
                # 在CPU上运行
                with torch.no_grad():
                    sam_features = cpu_encoder(cpu_x)
                
                # 将结果移回原设备
                sam_features = sam_features.to(device)
                
                return sam_features
            else:
                # 如果是其他错误，则重新抛出
                raise e


class SAMFeatureFusion(nn.Module):
    """
    SAM特征融合模块，用于将SAM特征与CLIP特征融合
    """
    def __init__(self, sam_dim=256, clip_dims=None, fusion_type="add"):
        """
        初始化SAM特征融合模块
        
        Args:
            sam_dim: SAM特征维度
            clip_dims: CLIP特征维度字典，如果为None则根据需要创建投影层
            fusion_type: 融合类型，可选"add"或"concat"
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.sam_dim = sam_dim
        
        # 创建投影层，用于将CLIP特征投影到SAM特征维度
        if clip_dims is not None:
            self.projections = nn.ModuleDict()
            for key, dim in clip_dims.items():
                self.projections[key] = nn.Conv2d(dim, sam_dim, kernel_size=1, bias=False)
    
    def forward(self, sam_features, clip_features):
        """
        前向传播
        
        Args:
            sam_features: SAM特征 [B, sam_dim, H, W]
            clip_features: CLIP特征字典 {layer_name: [B, dim, H, W]}
            
        Returns:
            fused_features: 融合后的特征字典 {layer_name: [B, dim, H, W]}
        """
        fused_features = {}
        
        for key, clip_feat in clip_features.items():
            try:
                # 如果CLIP特征维度与SAM特征维度不同，则进行投影
                if clip_feat.shape[1] != self.sam_dim and hasattr(self, 'projections'):
                    clip_feat_proj = self.projections[key](clip_feat)
                else:
                    clip_feat_proj = clip_feat
                
                # 如果特征尺寸不同，则将SAM特征上采样到CLIP特征尺寸
                if sam_features.shape[2:] != clip_feat_proj.shape[2:]:
                    sam_features_resized = F.interpolate(
                        sam_features, 
                        size=clip_feat_proj.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                else:
                    sam_features_resized = sam_features
                
                # 特征融合
                if self.fusion_type == "add":
                    fused_features[key] = clip_feat_proj + sam_features_resized
                elif self.fusion_type == "concat":
                    # 如果是拼接融合，需要创建新的投影层
                    if not hasattr(self, 'concat_projections'):
                        self.concat_projections = nn.ModuleDict()
                    
                    # 如果当前键的投影层不存在，则创建
                    if key not in self.concat_projections:
                        concat_dim = clip_feat_proj.shape[1] + sam_features_resized.shape[1]
                        out_dim = clip_feat_proj.shape[1]  # 保持输出维度与CLIP特征相同
                        self.concat_projections[key] = nn.Conv2d(concat_dim, out_dim, kernel_size=1, bias=False)
                        # 将投影层移动到与特征相同的设备
                        self.concat_projections[key] = self.concat_projections[key].to(clip_feat_proj.device)
                    
                    # 拼接特征
                    concat_features = torch.cat([clip_feat_proj, sam_features_resized], dim=1)
                    # 投影回原始维度
                    fused_features[key] = self.concat_projections[key](concat_features)
                else:
                    raise ValueError(f"不支持的融合类型: {self.fusion_type}")
            except Exception as e:
                print(f"警告: 特征融合失败: {e}")
                # 如果融合失败，则使用原始CLIP特征
                fused_features[key] = clip_feat
        
        return fused_features 