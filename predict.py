from typing import List, Union  # 导入List和Union类型，用于类型注解

import numpy as np  # 导入NumPy库，用于数值操作

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings  # 导入警告模块

    from shapely.errors import ShapelyDeprecationWarning  # 从shapely.errors导入ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)  # 忽略ShapelyDeprecationWarning
except:
    pass  # 如果导入失败，则忽略
import os  # 导入os模块，用于操作系统相关功能

import huggingface_hub  # 导入huggingface_hub库，用于与Hugging Face Hub交互
import torch  # 导入PyTorch库
from detectron2.checkpoint import DetectionCheckpointer  # 从detectron2导入DetectionCheckpointer，用于加载模型权重
from detectron2.config import get_cfg  # 从detectron2导入get_cfg，用于获取配置
from detectron2.data import MetadataCatalog  # 从detectron2导入MetadataCatalog，用于管理数据集元数据
from detectron2.engine import DefaultTrainer  # 从detectron2导入DefaultTrainer，用于构建模型
from detectron2.projects.deeplab import add_deeplab_config  # 从detectron2.projects.deeplab导入add_deeplab_config，用于添加DeepLab相关配置
from detectron2.utils.visualizer import Visualizer, random_color  # 从detectron2.utils.visualizer导入Visualizer和random_color，用于可视化
from huggingface_hub import hf_hub_download  # 从huggingface_hub导入hf_hub_download，用于从Hugging Face Hub下载文件
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理

from san import add_san_config  # 从san模块导入add_san_config，用于添加SAN相关配置
from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES  # 从san.data.datasets.register_coco_stuff_164k导入COCO_CATEGORIES

model_cfg = {  # 定义模型配置字典
    "san_vit_b_16": {  # SAN ViT-B/16模型配置
        "config_file": "configs/san_clip_vit_res4_coco.yaml",  # 配置文件路径
        "model_path": "huggingface:san_vit_b_16.pth",  # 模型路径，huggingface表示从Hugging Face Hub下载
    },
    "san_vit_large_16": {  # SAN ViT-L/16模型配置
        "config_file": "configs/san_clip_vit_large_res4_coco.yaml",  # 配置文件路径
        "model_path": "huggingface:san_vit_large_14.pth",  # 模型路径
    },
}


def download_model(model_path: str):  # 定义下载模型的函数
    """
    Download the model from huggingface hub.
    Args:
        model_path (str): the model path
    Returns:
        str: the downloaded model path
    """
    if "HF_TOKEN" in os.environ:  # 如果环境变量中存在HF_TOKEN
        huggingface_hub.login(token=os.environ["HF_TOKEN"])  # 使用HF_TOKEN登录Hugging Face Hub
    model_path = model_path.split(":")[1]  # 提取模型文件名
    model_path = hf_hub_download("Mendel192/san", filename=model_path)  # 从指定的仓库和文件名下载模型
    return model_path  # 返回下载后的模型路径


def setup(config_file: str, device=None):  # 定义设置配置的函数
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()  # 获取默认配置
    # for poly lr schedule
    add_deeplab_config(cfg)  # 添加DeepLab相关配置
    add_san_config(cfg)  # 添加SAN相关配置
    cfg.merge_from_file(config_file)  # 从配置文件加载配置
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"  # 设置模型运行设备，优先使用传入的device，否则根据CUDA是否可用选择cuda或cpu
    cfg.freeze()  # 冻结配置，防止后续修改
    return cfg  # 返回配置对象


class Predictor(object):  # 定义预测器类
    def __init__(self, config_file: str, model_path: str):  # 初始化方法
        """
        Args:
            config_file (str): the config file path
            model_path (str): the model path
        """
        cfg = setup(config_file)  # 设置配置
        self.model = DefaultTrainer.build_model(cfg)  # 根据配置构建模型
        if model_path.startswith("huggingface:"):  # 如果模型路径以"huggingface:"开头
            model_path = download_model(model_path)  # 下载模型
        print("Loading model from: ", model_path)  # 打印加载模型信息
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(  # 创建模型检查点对象并加载模型权重
            model_path
        )
        print("Loaded model from: ", model_path)  # 打印模型加载完成信息
        self.model.eval()  # 将模型设置为评估模式
        if torch.cuda.is_available():  # 如果CUDA可用
            self.device = torch.device("cuda")  # 设置设备为cuda
            self.model = self.model.cuda()  # 将模型移动到CUDA设备

    def predict(
        self,
        image_data_or_path: Union[Image.Image, str],  # 输入图像数据或路径
        vocabulary: List[str] = [],  # 词汇表，默认为空列表
        augment_vocabulary: Union[str,bool] = True,  # 是否增强词汇表，或指定增强方式，默认为True
        output_file: str = None,  # 输出文件路径，默认为None
    ) -> Union[dict, None]:  # 返回包含预测结果的字典或None
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
            augment_vocabulary (bool): whether to augment the vocabulary
            output_file (str): the output file path
        Returns:
            Union[dict, None]: the segmentation result
        """
        if isinstance(image_data_or_path, str):  # 如果输入是字符串路径
            image_data = Image.open(image_data_or_path)  # 打开图像文件
        else:  # 否则认为是图像数据
            image_data = image_data_or_path
        w, h = image_data.size  # 获取图像的宽度和高度
        image_tensor: torch.Tensor = self._preprocess(image_data)  # 预处理图像数据
        vocabulary = list(set([v.lower().strip() for v in vocabulary]))  # 将词汇表转换为小写并去除首尾空格，然后去重
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]  # 移除空字符串词汇
        print("vocabulary:", vocabulary)  # 打印当前使用的词汇表
        ori_vocabulary = vocabulary  # 保存原始词汇表

        if isinstance(augment_vocabulary,str):  # 如果augment_vocabulary是字符串（指定了增强方式）
            vocabulary = self.augment_vocabulary(vocabulary, augment_vocabulary)  # 增强词汇表
        else:  # 否则认为是布尔值
            vocabulary = self._merge_vocabulary(vocabulary)  # 合并词汇表（如果augment_vocabulary为True，则合并默认词汇）
        if len(ori_vocabulary) == 0:  # 如果原始词汇表为空
            ori_vocabulary = vocabulary  # 将处理后的词汇表作为原始词汇表
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            result = self.model(  # 模型前向传播
                [
                    {
                        "image": image_tensor,  # 输入图像张量
                        "height": h,  # 图像高度
                        "width": w,  # 图像宽度
                        "vocabulary": vocabulary,  # 使用的词汇表
                    }
                ]
            )[0]["sem_seg"]  # 获取语义分割结果
        seg_map = self._postprocess(result, ori_vocabulary)  # 后处理分割结果
        if output_file:  # 如果指定了输出文件路径
            self.visualize(image_data, seg_map, ori_vocabulary, output_file)  # 可视化结果并保存到文件
            return  # 返回None
        return {  # 返回包含结果的字典
            "image": image_data,  # 输入图像
            "sem_seg": seg_map,  # 分割图
            "vocabulary": ori_vocabulary,  # 原始词汇表
        }

    def visualize(
        self,
        image: Image.Image,  # 输入图像
        sem_seg: np.ndarray,  # 分割图
        vocabulary: List[str],  # 词汇表
        output_file: str = None,  # 输出文件路径，默认为None
        mode: str = "overlay",  # 可视化模式，"overlay"或"mask"，默认为"overlay"
    ) -> Union[Image.Image, None]:  # 返回可视化后的图像或None
        """
        Visualize the segmentation result.
        Args:
            image (Image.Image): the input image
            sem_seg (np.ndarray): the segmentation result
            vocabulary (List[str]): the vocabulary used for the segmentation
            output_file (str): the output file path
            mode (str): the visualization mode, can be "overlay" or "mask"
        Returns:
            Image.Image: the visualization result. If output_file is not None, return None.
        """
        # add temporary metadata
        # set numpy seed to make sure the colors are the same
        np.random.seed(0)  # 设置NumPy随机种子以确保颜色一致性
        colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]  # 为每个词汇生成随机颜色
        MetadataCatalog.get("_temp").set(stuff_classes=vocabulary, stuff_colors=colors)  # 创建临时元数据并设置类别和颜色
        metadata = MetadataCatalog.get("_temp")  # 获取临时元数据
        if mode == "overlay":  # 如果是叠加模式
            v = Visualizer(image, metadata)  # 创建Visualizer对象
            v = v.draw_sem_seg(sem_seg, area_threshold=0).get_image()  # 绘制语义分割结果并获取图像
            v = Image.fromarray(v)  # 将NumPy数组转换为PIL图像
        else:  # 否则是掩码模式
            v = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)  # 创建一个黑色背景图像
            labels, areas = np.unique(sem_seg, return_counts=True)  # 获取唯一的标签和它们的面积
            sorted_idxs = np.argsort(-areas).tolist()  # 根据面积降序排序标签索引
            labels = labels[sorted_idxs]  # 获取排序后的标签
            for label in filter(lambda l: l < len(metadata.stuff_classes), labels):  # 遍历有效标签
                v[sem_seg == label] = metadata.stuff_colors[label]  # 将对应区域填充颜色
            v = Image.fromarray(v)  # 将NumPy数组转换为PIL图像
        # remove temporary metadata
        MetadataCatalog.remove("_temp")  # 移除临时元数据
        if output_file is None:  # 如果未指定输出文件
            return v  # 返回可视化图像
        v.save(output_file)  # 保存可视化图像到文件
        print(f"saved to {output_file}")  # 打印保存信息

    def _merge_vocabulary(self, vocabulary: List[str]) -> List[str]:  # 定义合并词汇表的私有方法
        default_voc = [c["name"] for c in COCO_CATEGORIES]  # 获取COCO数据集的默认词汇
        return vocabulary + [c for c in default_voc if c not in vocabulary]  # 合并输入词汇和不在输入词汇中的默认词汇

    def augment_vocabulary(
        self, vocabulary: List[str], aug_set: str = "COCO-all"  # 增强词汇表的私有方法，aug_set指定增强集
    ) -> List[str]:
        default_voc = [c["name"] for c in COCO_CATEGORIES]  # 获取COCO数据集的默认词汇
        stuff_voc = [  # 获取COCO数据集中"stuff"类别（背景类）的词汇
            c["name"]
            for c in COCO_CATEGORIES
            if "isthing" not in c or c["isthing"] == 0
        ]
        if aug_set == "COCO-all":  # 如果增强集为"COCO-all"
            return vocabulary + [c for c in default_voc if c not in vocabulary]  # 合并输入词汇和不在输入词汇中的COCO所有词汇
        elif aug_set == "COCO-stuff":  # 如果增强集为"COCO-stuff"
            return vocabulary + [c for c in stuff_voc if c not in vocabulary]  # 合并输入词汇和不在输入词汇中的COCO stuff词汇
        else:
            return vocabulary  # 否则返回原始词汇

    def _preprocess(self, image: Image.Image) -> torch.Tensor:  # 定义图像预处理的私有方法
        """
        Preprocess the input image.
        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the preprocessed image
        """
        image = image.convert("RGB")  # 将图像转换为RGB格式
        # resize short side to 640
        w, h = image.size  # 获取图像宽度和高度
        if w < h:  # 如果宽度小于高度
            image = image.resize((640, int(h * 640 / w)))  # 将短边（宽度）缩放到640，并按比例缩放高度
        else:  # 否则（高度小于等于宽度）
            image = image.resize((int(w * 640 / h), 640))  # 将短边（高度）缩放到640，并按比例缩放宽度
        image = torch.from_numpy(np.asarray(image)).float()  # 将PIL图像转换为NumPy数组，再转换为PyTorch浮点张量
        image = image.permute(2, 0, 1)  # 调整维度顺序 (H, W, C) -> (C, H, W)
        return image  # 返回预处理后的图像张量

    def _postprocess(
        self, result: torch.Tensor, ori_vocabulary: List[str]  # 定义后处理分割结果的私有方法
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = result.argmax(dim=0).cpu().numpy()  # (H, W) # 在类别维度上取最大值索引，转换为CPU上的NumPy数组
        if len(ori_vocabulary) == 0:  # 如果原始词汇表为空
            return result  # 直接返回结果
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)  # 将超出原始词汇表长度的预测标签设置为词汇表长度（表示背景或其他）
        return result  # 返回后处理后的分割图


def pre_download():  # 定义预下载模型的函数
    """pre downlaod model from huggingface and open_clip to avoid network issue."""
    for model_name, model_info in model_cfg.items():  # 遍历模型配置
        download_model(model_info["model_path"])  # 下载模型权重
        cfg = setup(model_info["config_file"])  # 设置配置
        DefaultTrainer.build_model(cfg)  # 构建模型（可能会触发open_clip相关下载）


if __name__ == "__main__":  # 如果作为主脚本运行
    from argparse import ArgumentParser  # 导入ArgumentParser用于解析命令行参数

    parser = ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument(  # 添加配置文件参数
        "--config-file", type=str, required=True, help="path to config file"
    )
    parser.add_argument(  # 添加模型路径参数
        "--model-path", type=str, required=True, help="path to model file"
    )
    parser.add_argument(  # 添加图像路径参数
        "--img-path", type=str, required=True, help="path to image file."
    )
    parser.add_argument("--aug-vocab", action="store_true", help="augment vocabulary.")  # 添加是否增强词汇表的标志参数
    parser.add_argument(  # 添加词汇表参数
        "--vocab",
        type=str,
        default="",
        help="list of category name. seperated with ,.",
    )
    parser.add_argument(  # 添加输出文件路径参数
        "--output-file", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()  # 解析命令行参数
    predictor = Predictor(config_file=args.config_file, model_path=args.model_path)  # 创建预测器对象
    predictor.predict(  # 执行预测
        args.img_path,  # 图像路径
        args.vocab.split(","),  # 词汇表（按逗号分割）
        args.aug_vocab,  # 是否增强词汇表
        output_file=args.output_file,  # 输出文件路径
    )
