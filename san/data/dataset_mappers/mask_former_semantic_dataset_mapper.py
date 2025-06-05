import copy  # 导入copy模块，用于深拷贝
import logging  # 导入日志模块

import numpy as np  # 导入NumPy库，用于数值操作
import torch  # 导入PyTorch库
from torch.nn import functional as F  # 导入PyTorch的函数式接口

from detectron2.config import configurable  # 从detectron2导入可配置装饰器
from detectron2.data import MetadataCatalog  # 从detectron2导入元数据目录
from detectron2.data import detection_utils as utils  # 从detectron2导入检测工具函数
from detectron2.data import transforms as T  # 从detectron2导入数据增强变换
from detectron2.projects.point_rend import ColorAugSSDTransform  # 从PointRend项目导入颜色增强变换
from detectron2.structures import BitMasks, Instances  # 从detectron2导入BitMasks和Instances结构

__all__ = ["MaskFormerSemanticDatasetMapper"]  # 定义公开接口


class MaskFormerSemanticDatasetMapper:
    """
    一个可调用对象，接收Detectron2数据集格式的字典，
    并将其映射为MaskFormer用于语义分割的格式。

    该可调用对象当前执行以下操作：

    1. 从"file_name"读取图像
    2. 对图像和标注应用几何变换
    3. 对图像和标注查找并应用合适的裁剪
    4. 将图像和标注准备为张量
    """

    @configurable  # 使该类可以通过配置进行实例化
    def __init__(
        self,
        is_train=True,  # 是否为训练模式
        *,
        augmentations,  # 数据增强操作列表
        image_format,  # 图像格式
        ignore_label,  # 评估时忽略的标签
        size_divisibility,  # 图像尺寸的可除性因子
    ):
        """
        注意：此接口是实验性的。
        参数:
            is_train: 用于训练还是推理
            augmentations: 要应用的一系列增强或确定性变换
            image_format: :func:`detection_utils.read_image` 支持的图像格式。
            ignore_label: 评估时忽略的标签
            size_divisibility: 将图像尺寸填充到可被该值整除
        """
        self.is_train = is_train  # 设置训练模式标志
        self.tfm_gens = augmentations  # 存储数据增强生成器
        self.img_format = image_format  # 存储图像格式
        self.ignore_label = ignore_label  # 存储忽略标签
        self.size_divisibility = size_divisibility  # 存储尺寸可除性因子

        logger = logging.getLogger(__name__)  # 获取日志记录器
        mode = "training" if is_train else "inference"  # 根据is_train设置模式字符串
        logger.info(
            f"[{self.__class__.__name__}] 在 {mode} 中使用的增强: {augmentations}"
        )  # 记录使用的增强信息

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # 从配置构建数据增强
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,  # 训练时最短边最小尺寸
                cfg.INPUT.MAX_SIZE_TRAIN,  # 训练时最大尺寸
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,  # 训练时最短边采样方式
            )
        ]  # 基础的调整大小增强
        if cfg.INPUT.CROP.ENABLED:  # 如果启用了裁剪
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,  # 裁剪类型
                    cfg.INPUT.CROP.SIZE,  # 裁剪尺寸
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,  # 单类别最大区域限制
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,  # 忽略值
                )
            )  # 添加带类别区域约束的随机裁剪
        if cfg.INPUT.COLOR_AUG_SSD:  # 如果启用了SSD颜色增强
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))  # 添加SSD颜色增强
        augs.append(T.RandomFlip())  # 添加随机翻转

        # 假设总是应用于训练集
        dataset_names = cfg.DATASETS.TRAIN  # 获取训练数据集名称
        meta = MetadataCatalog.get(dataset_names[0])  # 获取第一个训练数据集的元数据
        ignore_label = meta.ignore_label  # 从元数据中获取忽略标签

        ret = {
            "is_train": is_train,  # 是否为训练模式
            "augmentations": augs,  # 数据增强列表
            "image_format": cfg.INPUT.FORMAT,  # 图像格式
            "ignore_label": ignore_label,  # 忽略标签
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,  # 尺寸可除性因子
        }
        return ret  # 返回构建的参数字典

    def __call__(self, dataset_dict):
        """
        参数:
            dataset_dict (dict): 单个图像的元数据，采用Detectron2数据集格式。

        返回:
            dict: detectron2内置模型接受的格式
        """
        assert (
            self.is_train
        ), "MaskFormerSemanticDatasetMapper应该仅用于训练！"  # 确保仅在训练时使用

        dataset_dict = copy.deepcopy(dataset_dict)  # 深拷贝数据集字典，因为后续代码会修改它
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)  # 读取图像
        utils.check_image_size(dataset_dict, image)  # 检查图像尺寸

        if "sem_seg_file_name" in dataset_dict:  # 如果存在语义分割标签文件名
            # PyTorch变换未对uint16实现，因此先将其转换为double
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype(
                "double"
            )  # 读取语义分割真值并转换为double类型
        else:
            sem_seg_gt = None  # 否则，语义分割真值为None

        if sem_seg_gt is None:  # 如果语义分割真值为None
            raise ValueError(
                "在语义分割数据集 {} 中找不到 'sem_seg_file_name'。".format(
                    dataset_dict["file_name"]
                )
            )  # 抛出值错误

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)  # 创建数据增强输入对象
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)  # 应用数据增强
        image = aug_input.image  # 获取增强后的图像
        sem_seg_gt = aug_input.sem_seg  # 获取增强后的语义分割真值

        # 在此处填充图像和分割标签！
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))  # 将图像转换为张量并调整维度顺序 (H,W,C) -> (C,H,W)
        if sem_seg_gt is not None:  # 如果存在语义分割真值
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))  # 将语义分割真值转换为长整型张量

        if self.size_divisibility > 0:  # 如果尺寸可除性因子大于0
            image_size = (image.shape[-2], image.shape[-1])  # 获取图像尺寸 (H,W)
            padding_size = [
                0,  # 左填充
                self.size_divisibility - image_size[1] % self.size_divisibility if image_size[1] % self.size_divisibility != 0 else 0, # 右填充
                0,  # 上填充
                self.size_divisibility - image_size[0] % self.size_divisibility if image_size[0] % self.size_divisibility != 0 else 0, # 下填充
            ]  # 计算填充大小，使其能被size_divisibility整除
            image = F.pad(image, padding_size, value=128).contiguous()  # 填充图像，使用128作为填充值
            if sem_seg_gt is not None:  # 如果存在语义分割真值
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()  # 填充语义分割真值，使用ignore_label作为填充值

        image_shape = (image.shape[-2], image.shape[-1])  # 获取填充后图像的形状 (H, W)

        # PyTorch的dataloader在处理torch.Tensor时由于共享内存而高效，
        # 但在处理大型通用数据结构时由于使用pickle和mp.Queue而效率不高。
        # 因此，使用torch.Tensor非常重要。
        dataset_dict["image"] = image  # 将处理后的图像存入数据集字典

        if sem_seg_gt is not None:  # 如果存在语义分割真值
            dataset_dict["sem_seg"] = sem_seg_gt.long()  # 将语义分割真值（长整型）存入数据集字典

        if "annotations" in dataset_dict:  # 如果数据集中仍存在"annotations"键
            raise ValueError(
                "语义分割数据集不应包含 'annotations'。"
            )  # 抛出值错误，语义分割任务通常不直接使用实例级标注

        # 准备每个类别的二值掩码
        if sem_seg_gt is not None:  # 如果存在语义分割真值
            sem_seg_gt = sem_seg_gt.numpy()  # 将语义分割真值转换为NumPy数组
            instances = Instances(image_shape)  # 创建Instances对象
            classes = np.unique(sem_seg_gt)  # 获取语义分割真值中所有唯一的类别ID
            # 移除忽略区域
            classes = classes[classes != self.ignore_label]  # 从类别列表中移除忽略标签
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)  # 将有效类别ID存储为gt_classes

            masks = []  # 初始化掩码列表
            for class_id in classes:  # 遍历每个有效类别ID
                masks.append(sem_seg_gt == class_id)  # 为该类别创建二值掩码

            if len(masks) == 0:  # 如果图像中没有有效标注（所有区域都被忽略）
                # 某些图像没有标注（全部被忽略）
                instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
                )  # 创建一个空的掩码张量
            else:
                masks = BitMasks(
                    torch.stack(
                        [
                            torch.from_numpy(np.ascontiguousarray(x.copy()))
                            for x in masks
                        ]  # 将每个NumPy掩码转换为张量，并确保内存连续
                    )
                )  # 使用BitMasks结构存储所有类别的二值掩码
                instances.gt_masks = masks.tensor  # 将掩码张量存储到Instances对象中

            dataset_dict["instances"] = instances  # 将Instances对象（包含类别和掩码）存入数据集字典

        return dataset_dict  # 返回处理后的数据集字典