import itertools  # 导入itertools模块，用于高效循环
import logging  # 导入日志模块
import numpy as np  # 导入NumPy库
from collections import Counter  # 从collections模块导入Counter，用于计数
import torch.utils.data  # 导入PyTorch数据工具
from tabulate import tabulate  # 导入tabulate库，用于创建漂亮的表格
from termcolor import colored  # 导入termcolor库，用于彩色文本输出

from detectron2.utils.logger import _log_api_usage, log_first_n  # 从detectron2导入日志工具
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog  # 从detectron2导入数据集和元数据目录
import torch.utils.data  # 再次导入PyTorch数据工具（冗余）
from detectron2.config import configurable  # 从detectron2导入可配置装饰器
from detectron2.data.build import (
    build_batch_data_loader,  # 导入构建批量数据加载器的函数
    trivial_batch_collator,  # 导入简单的批处理整理函数
    load_proposals_into_dataset,  # 导入将提议加载到数据集的函数
    filter_images_with_only_crowd_annotations,  # 导入过滤仅含人群标注图像的函数
    filter_images_with_few_keypoints,  # 导入过滤关键点过少图像的函数
    print_instances_class_histogram,  # 导入打印实例类别直方图的函数
)

from detectron2.data.common import DatasetFromList, MapDataset  # 从detectron2导入通用数据类
from detectron2.data.dataset_mapper import DatasetMapper  # 从detectron2导入数据集映射器
from detectron2.data.detection_utils import check_metadata_consistency  # 从detectron2导入元数据一致性检查函数
from detectron2.data.samplers import (
    InferenceSampler,  # 推理采样器
    RandomSubsetTrainingSampler,  # 随机子集训练采样器
    RepeatFactorTrainingSampler,  # 重复因子训练采样器
    TrainingSampler,  # 训练采样器
)

"""
此文件包含构建训练或测试数据加载器的默认逻辑。
"""

__all__ = [
    "build_detection_train_loader",  # 构建检测训练数据加载器
    "build_detection_test_loader",  # 构建检测测试数据加载器
]


def print_classification_instances_class_histogram(dataset_dicts, class_names):
    """
    打印分类任务中实例的类别直方图。
    参数:
        dataset_dicts (list[dict]): 数据集字典列表。
        class_names (list[str]): 类别名称列表（从零开始索引）。
    """
    num_classes = len(class_names)  # 获取类别数量
    hist_bins = np.arange(num_classes + 1)  # 创建直方图的bins
    histogram = np.zeros((num_classes,), dtype=np.int)  # 初始化直方图
    for entry in dataset_dicts:  # 遍历数据集条目
        classes = np.asarray([entry["category_id"]], dtype=np.int)  # 获取类别ID
        if len(classes):  # 如果存在类别
            assert classes.min() >= 0, f"得到无效的category_id={classes.min()}"  # 检查类别ID有效性
            assert (
                classes.max() < num_classes
            ), f"对于{num_classes}个类别的数据集，得到无效的category_id={classes.max()}"  # 检查类别ID有效性
        histogram += np.histogram(classes, bins=hist_bins)[0]  # 更新直方图

    N_COLS = min(6, len(class_names) * 2)  # 设置表格的列数

    def short_name(x):
        # 缩短长类别名称。对lvis等数据集有用
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(
            *[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)]
        )
    )  # 准备表格数据
    total_num_instances = sum(data[1::2])  # 计算总实例数
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))  # 填充数据以适应表格列数
    if num_classes > 1:  # 如果类别数大于1
        data.extend(["total", total_num_instances])  # 添加总计行
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])  # 按列组织数据
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),  # 设置表头
        tablefmt="pipe",  # 设置表格格式
        numalign="left",  # 数字左对齐
        stralign="center",  # 字符串居中对齐
    )  # 生成表格
    log_first_n(
        logging.INFO,
        "所有{}个类别中实例的分布情况:\n".format(num_classes)
        + colored(table, "cyan"),  # 使用青色打印表格
        key="message",
    )  # 记录日志


def wrap_metas(dataset_dict, **kwargs):
    """
    将元数据包装到数据集字典的每个样本中。
    """
    def _assign_attr(data_dict: dict, **kwargs):
        # 确保分配的属性在原始样本中不存在。
        assert not any(
            [key in data_dict for key in kwargs]
        ), "分配的属性不应存在于原始样本中。"
        data_dict.update(kwargs)  # 更新字典
        return data_dict

    return [_assign_attr(sample, meta=kwargs) for sample in dataset_dict]  # 对每个样本应用属性分配


def get_detection_dataset_dicts(
    names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    """
    加载并准备用于实例检测/分割和语义分割的数据集字典。

    参数:
        names (str or list[str]): 数据集名称或名称列表
        filter_empty (bool): 是否过滤掉没有实例标注的图像
        min_keypoints (int): 过滤掉关键点少于`min_keypoints`的图像。设为0则不执行任何操作。
        proposal_files (list[str]): 如果给定，则为与`names`中每个数据集匹配的对象提议文件列表。

    返回:
        list[dict]: 符合标准数据集字典格式的字典列表。
    """
    if isinstance(names, str):  # 如果名称是字符串
        names = [names]  # 转换为列表
    assert len(names), names  # 确保名称列表不为空
    dataset_dicts = [
        wrap_metas(DatasetCatalog.get(dataset_name), dataset_name=dataset_name)
        for dataset_name in names
    ]  # 从DatasetCatalog获取数据集字典并包装元数据
    for dataset_name, dicts in zip(names, dataset_dicts):  # 遍历数据集名称和字典
        assert len(dicts), "数据集'{}'为空！".format(dataset_name)  # 确保数据集非空

    if proposal_files is not None:  # 如果提供了提议文件
        assert len(names) == len(proposal_files)  # 确保名称和提议文件数量匹配
        # 从提议文件加载预计算的提议
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))  # 将嵌套列表展平

    has_instances = "annotations" in dataset_dicts[0]  # 检查是否存在实例标注
    if filter_empty and has_instances:  # 如果过滤空标注且存在实例
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)  # 过滤仅含人群标注的图像
    if min_keypoints > 0 and has_instances:  # 如果设置了最小关键点数且存在实例
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)  # 过滤关键点过少的图像

    if has_instances:  # 如果存在实例
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes  # 获取物体类别名称
            check_metadata_consistency("thing_classes", names)  # 检查元数据一致性
            print_instances_class_histogram(dataset_dicts, class_names)  # 打印实例类别直方图
        except AttributeError:  # 此数据集的类别名称不可用
            pass  # 忽略错误

    assert len(dataset_dicts), "在{}中未找到有效数据。".format(",".join(names))  # 确保找到有效数据
    return dataset_dicts  # 返回数据集字典


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    # 从配置构建训练数据加载器
    if dataset is None:  # 如果未提供数据集
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,  # 训练数据集名称
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,  # 是否过滤空标注
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,  # 最小关键点数
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,  # 提议文件
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])  # 记录API使用情况

    if mapper is None:  # 如果未提供映射器
        mapper = DatasetMapper(cfg, True)  # 创建默认数据集映射器

    if sampler is None:  # 如果未提供采样器
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN  # 获取采样器名称
        logger = logging.getLogger(__name__)  # 获取日志记录器
        logger.info("使用训练采样器 {}".format(sampler_name))  # 记录使用的采样器
        if sampler_name == "TrainingSampler":  # 如果是训练采样器
            sampler = TrainingSampler(len(dataset))  # 创建训练采样器
        elif sampler_name == "RepeatFactorTrainingSampler":  # 如果是重复因子训练采样器
            repeat_factors = (
                RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
            )  # 计算重复因子
            sampler = RepeatFactorTrainingSampler(repeat_factors)  # 创建重复因子训练采样器
        elif sampler_name == "RandomSubsetTrainingSampler":  # 如果是随机子集训练采样器
            sampler = RandomSubsetTrainingSampler(
                len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
            )  # 创建随机子集训练采样器
        else:
            raise ValueError("未知的训练采样器: {}".format(sampler_name))  # 抛出错误

    return {
        "dataset": dataset,  # 数据集
        "sampler": sampler,  # 采样器
        "mapper": mapper,  # 映射器
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,  # 总批次大小
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,  # 是否进行宽高比分组
        "num_workers": cfg.DATALOADER.NUM_WORKERS,  # 工作进程数
    }


# TODO 可以允许dataset作为可迭代对象或IterableDataset，以使此函数更通用
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
):
    """
    构建具有某些默认功能的对象检测数据加载器。
    此接口是实验性的。

    参数:
        dataset (list or torch.utils.data.Dataset): 数据集字典列表，
            或映射式pytorch数据集。它们可以通过使用
            :func:`DatasetCatalog.get` 或 :func:`get_detection_dataset_dicts` 获得。
        mapper (callable): 一个可调用对象，它接受来自数据集的样本（字典）并
            返回模型要使用的格式。
            使用cfg时，默认选择是 ``DatasetMapper(cfg, is_train=True)``。
        sampler (torch.utils.data.sampler.Sampler or None): 应用于 ``dataset`` 的
            索引的采样器。默认为 :class:`TrainingSampler`，
            它协调所有工作进程之间的无限随机洗牌序列。
        total_batch_size (int): 所有工作进程的总批次大小。批处理
            只是将数据放入列表中。
        aspect_ratio_grouping (bool): 是否为提高效率而对具有相似
            宽高比的图像进行分组。启用时，它要求
            数据集中的每个元素都是具有键“width”和“height”的字典。
        num_workers (int): 并行数据加载工作进程的数量

    返回:
        torch.utils.data.DataLoader:
            一个数据加载器。它的每个输出都是长度为
            ``total_batch_size / num_workers`` 的 ``list[mapped_element]``，
            其中 ``mapped_element`` 由 ``mapper`` 生成。
    """
    if isinstance(dataset, list):  # 如果数据集是列表
        dataset = DatasetFromList(dataset, copy=False)  # 从列表创建数据集对象
    if mapper is not None:  # 如果提供了映射器
        dataset = MapDataset(dataset, mapper)  # 应用映射器到数据集
    if sampler is None:  # 如果未提供采样器
        sampler = TrainingSampler(len(dataset))  # 创建默认训练采样器
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)  # 确保采样器是Sampler类型
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,  # 是否进行宽高比分组
        num_workers=num_workers,  # 工作进程数
    )


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    使用给定的`dataset_name`参数（而不是cfg中的名称），因为
    标准做法是单独评估每个测试集（而不是组合它们）。
    """
    if isinstance(dataset_name, str):  # 如果数据集名称是字符串
        dataset_name = [dataset_name]  # 转换为列表

    dataset = get_detection_dataset_dicts(
        dataset_name,  # 数据集名称
        filter_empty=False,  # 不过滤空标注
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
            for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,  # 提议文件
    )
    if mapper is None:  # 如果未提供映射器
        mapper = DatasetMapper(cfg, False)  # 创建默认数据集映射器（用于测试）
    return {
        "dataset": dataset,  # 数据集
        "mapper": mapper,  # 映射器
        "num_workers": cfg.DATALOADER.NUM_WORKERS,  # 工作进程数
        "samples_per_gpu": cfg.SOLVER.TEST_IMS_PER_BATCH,  # 每个GPU的样本数
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset, *, mapper, sampler=None, num_workers=0, samples_per_gpu=1
):
    """
    与`build_detection_train_loader`类似，但使用批大小为1，
    和 :class:`InferenceSampler`。此采样器协调所有工作进程
    以生成所有样本的精确集合。
    此接口是实验性的。

    参数:
        dataset (list or torch.utils.data.Dataset): 数据集字典列表，
            或映射式pytorch数据集。它们可以通过使用
            :func:`DatasetCatalog.get` 或 :func:`get_detection_dataset_dicts` 获得。
        mapper (callable): 一个可调用对象，它接受来自数据集的样本（字典）
           并返回模型要使用的格式。
           使用cfg时，默认选择是 ``DatasetMapper(cfg, is_train=False)``。
        sampler (torch.utils.data.sampler.Sampler or None): 应用于 ``dataset`` 的
            索引的采样器。默认为 :class:`InferenceSampler`，
            它在所有工作进程之间分割数据集。
        num_workers (int): 并行数据加载工作进程的数量

    返回:
        DataLoader: 一个torch DataLoader，加载给定的检测
        数据集，具有测试时转换和批处理功能。

    示例:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # 或者，使用CfgNode实例化：
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):  # 如果数据集是列表
        dataset = DatasetFromList(dataset, copy=False)  # 从列表创建数据集对象
    if mapper is not None:  # 如果提供了映射器
        dataset = MapDataset(dataset, mapper)  # 应用映射器到数据集
    if sampler is None:  # 如果未提供采样器
        sampler = InferenceSampler(len(dataset))  # 创建推理采样器
    # 推理期间始终每个工作进程使用1个图像，因为这是
    # 论文中报告推理时间的标准。
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, samples_per_gpu, drop_last=False
    )  # 创建批处理采样器
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,  # 工作进程数
        batch_sampler=batch_sampler,  # 批处理采样器
        collate_fn=trivial_batch_collator,  # 整理函数
    )
    return data_loader  # 返回数据加载器


def dataset_sample_per_class(cfg):
    # 每个类别采样数据集
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,  # 训练数据集名称
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,  # 是否过滤空标注
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,  # 最小关键点数
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,  # 提议文件
    )
    if cfg.DATASETS.SAMPLE_PER_CLASS > 0:  # 如果设置了每个类别的样本数
        category_list = [data["category_id"] for data in dataset_dicts]  # 获取类别ID列表
        category_count = Counter(category_list)  # 统计每个类别的数量
        category_group = {
            cat: [data for data in dataset_dicts if data["category_id"] == cat]
            for cat in category_count.keys()
        }  # 按类别分组数据
        rng = np.random.default_rng(cfg.DATASETS.SAMPLE_SEED)  # 创建随机数生成器
        selected = {
            cat: groups
            if len(groups) < cfg.DATASETS.SAMPLE_PER_CLASS  # 如果组内样本数小于设定值
            else rng.choice(groups, size=cfg.DATASETS.SAMPLE_PER_CLASS).tolist()  # 随机选择样本
            for cat, groups in category_group.items()  # 遍历类别和组
        }  # 选择样本
        tmp = []  # 临时列表
        for k, v in selected.items():  # 遍历选择的样本
            tmp.extend(v)  # 添加到临时列表
        dataset_dicts = tmp  # 更新数据集字典
        logger = logging.getLogger(__name__)  # 获取日志记录器
        logger.info(tmp)  # 记录选择的样本
    dataset = dataset_dicts  # 设置数据集
    print_classification_instances_class_histogram(
        dataset, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
    )  # 打印分类实例类别直方图
    return dataset  # 返回数据集