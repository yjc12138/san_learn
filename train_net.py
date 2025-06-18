try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings  # 导入警告模块

    from shapely.errors import ShapelyDeprecationWarning  # 从shapely.errors导入ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)  # 忽略ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")  # 忽略特定用户警告
    warnings.filterwarnings("ignore", category=FutureWarning) 
except:
    pass  # 如果导入失败，则忽略
import copy  # 导入copy模块，用于深拷贝
import itertools  # 导入itertools模块，用于创建迭代器
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于操作系统相关功能
# 设置环境变量
os.environ["DETECTRON2_DATASETS"] = "/home/Tarkiya/project/NLP/code/yjc/data"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,3,2,4"
os.environ["WANDB_MODE"] = "offline"
from collections import OrderedDict, defaultdict  # 从collections导入OrderedDict和defaultdict
from typing import Any, Dict, List, Set  # 导入类型注解

import detectron2.utils.comm as comm  # 导入detectron2的通信工具
import torch  # 导入PyTorch库

from detectron2.checkpoint import DetectionCheckpointer  # 从detectron2导入模型检查点工具
from detectron2.config import get_cfg  # 从detectron2导入获取配置的函数
from detectron2.data import MetadataCatalog  # 从detectron2导入元数据目录
from detectron2.engine import (  # 从detectron2.engine导入常用组件
    DefaultTrainer,  # 默认训练器类
    default_argument_parser,  # 默认参数解析器
    default_setup,  # 默认设置函数
    launch,  # 启动分布式训练的函数
)
from detectron2.evaluation import (  # 从detectron2.evaluation导入评估相关组件
    CityscapesSemSegEvaluator,  # Cityscapes语义分割评估器
    DatasetEvaluators,  # 数据集评估器集合
    SemSegEvaluator,  # 语义分割评估器
    verify_results,  # 验证结果的函数
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler  # 从DeepLab项目导入配置和学习率调度器构建函数
from detectron2.solver.build import maybe_add_gradient_clipping  # 从detectron2.solver.build导入可能添加梯度裁剪的函数
from detectron2.utils.logger import setup_logger  # 从detectron2.utils.logger导入日志设置函数
from tabulate import tabulate  # 导入tabulate库，用于格式化表格输出

from san import (  # 从san模块导入自定义组件
    MaskFormerSemanticDatasetMapper,  # MaskFormer语义分割数据集映射器
    SemanticSegmentorWithTTA,  # 带测试时增强的语义分割器
    add_san_config,  # 添加SAN配置的函数
)
from san.data import build_detection_test_loader, build_detection_train_loader  # 从san.data导入数据加载器构建函数
from san.utils import WandbWriter, setup_wandb  # 从san.utils导入Wandb写入器和设置函数


class Trainer(DefaultTrainer):  # 定义自定义训练器类，继承自DefaultTrainer
# 写入器（Writer）是用于记录训练过程中各种指标和数据的组件，主要功能包括：
# 记录训练损失、准确率等指标
# 保存模型检查点
# 可视化训练过程（如损失曲线、学习率变化等）
# 输出日志信息
    def build_writers(self):  # 重写构建写入器的方法
        writers = super().build_writers()  # 调用父类的构建写入器方法
        # use wandb writer instead.
        writers[-1] = WandbWriter()  # 将最后一个写入器替换为WandbWriter，最后一个通常负责可视化功能
        return writers  # 返回写入器列表

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):  # 重写构建评估器的方法
        if output_folder is None:  # 如果未指定输出文件夹
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")  # 设置默认输出文件夹
        evaluator_list = []  # 初始化评估器列表
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type  # 获取数据集的评估器类型
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:  # 如果是语义分割或ADE20K全景分割类型
            evaluator_list.append(  # 添加SemSegEvaluator
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,  # 启用分布式评估
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "cityscapes_sem_seg":  # 如果是Cityscapes语义分割类型
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."  # 断言CityscapesEvaluator在多机情况下可能不工作
            return CityscapesSemSegEvaluator(dataset_name)  # 返回CityscapesSemSegEvaluator

        if len(evaluator_list) == 0:  # 如果评估器列表为空
            raise NotImplementedError(  # 抛出未实现错误
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:  # 如果评估器列表只有一个元素
            return evaluator_list[0]  # 返回该评估器
        return DatasetEvaluators(evaluator_list)  # 返回包含多个评估器的DatasetEvaluators

    @classmethod
    def build_train_loader(cls, cfg):  # 重写构建训练数据加载器的方法
        # resue maskformer dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":  # 如果数据集映射器名称为"mask_former_semantic"
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)  # 创建MaskFormerSemanticDatasetMapper
            return build_detection_train_loader(cfg, mapper=mapper)  # 使用自定义mapper构建训练数据加载器
        else:
            mapper = None  # 否则不使用特定mapper
            return build_detection_train_loader(cfg, mapper=mapper)  # 构建训练数据加载器

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):  # 重写构建测试数据加载器的方法
        # Add dataset meta info.
        return build_detection_test_loader(cfg, dataset_name)  # 构建测试数据加载器

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):  # 重写构建学习率调度器的方法
        # use poly scheduler
        return build_lr_scheduler(cfg, optimizer)  # 使用DeepLab的build_lr_scheduler构建学习率调度器

    @classmethod
    def build_optimizer(cls, cfg, model):  # 重写构建优化器的方法
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM  # 获取归一化层权重衰减系数
        weight_decay_embed_group = cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP  # 获取嵌入层组权重衰减的参数名列表
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED  # 获取嵌入层权重衰减系数

        defaults = {}  # 初始化默认参数字典
        defaults["lr"] = cfg.SOLVER.BASE_LR  # 设置默认学习率
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY  # 设置默认权重衰减

        norm_module_types = (  # 定义归一化模块类型元组
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []  # 初始化参数列表，用于存储不同参数组的配置
        memo: Set[torch.nn.parameter.Parameter] = set()  # 初始化集合，用于记录已处理的参数，避免重复
        for module_name, module in model.named_modules():  # 遍历模型的所有模块
            for module_param_name, value in module.named_parameters(recurse=False):  # 遍历模块的直接参数
                if not value.requires_grad:  # 如果参数不需要梯度，则跳过
                    continue
                # Avoid duplicating parameters
                if value in memo:  # 如果参数已处理，则跳过
                    continue
                memo.add(value)  # 将参数添加到已处理集合

                hyperparams = copy.copy(defaults)  # 复制默认超参数
                hyperparams["param_name"] = ".".join([module_name, module_param_name])  # 记录参数名称
                if "side_adapter_network" in module_name:  # 如果模块名包含"side_adapter_network"
                    hyperparams["lr"] = (  # 调整学习率
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                # scale clip lr
                if "clip" in module_name:  # 如果模块名包含"clip"
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER  # 调整学习率
                if any([x in module_param_name for x in weight_decay_embed_group]):  # 如果参数名在嵌入层组权重衰减列表中
                    hyperparams["weight_decay"] = weight_decay_embed  # 设置嵌入层权重衰减
                if isinstance(module, norm_module_types):  # 如果模块是归一化类型
                    hyperparams["weight_decay"] = weight_decay_norm  # 设置归一化层权重衰减
                if isinstance(module, torch.nn.Embedding):  # 如果模块是嵌入层
                    hyperparams["weight_decay"] = weight_decay_embed  # 设置嵌入层权重衰减
                params.append({"params": [value], **hyperparams})  # 将参数及其超参数添加到参数列表

        def maybe_add_full_model_gradient_clipping(optim):  # 定义可能添加全模型梯度裁剪的函数
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE  # 获取梯度裁剪值
            enable = (  # 判断是否启用全模型梯度裁剪
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):  # 定义带全模型梯度裁剪的优化器类
                def step(self, closure=None):  # 重写step方法
                    all_params = itertools.chain(  # 获取所有参数
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)  # 进行梯度裁剪
                    super().step(closure=closure)  # 调用父类的step方法

            return FullModelGradientClippingOptimizer if enable else optim  # 如果启用则返回带裁剪的优化器，否则返回原优化器

        optimizer_type = cfg.SOLVER.OPTIMIZER  # 获取优化器类型
        if optimizer_type == "SGD":  # 如果是SGD优化器
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(  # 创建SGD优化器
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":  # 如果是AdamW优化器
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(  # 创建AdamW优化器
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")  # 如果优化器类型不支持则抛出错误
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":  # 如果不是全模型梯度裁剪（即按参数组裁剪）
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)  # 添加按参数组的梯度裁剪
        # display the lr and wd of each param group in a table
        optim_info = defaultdict(list)  # 初始化优化器信息字典
        total_params_size = 0  # 初始化总参数量
        for group in optimizer.param_groups:  # 遍历优化器的参数组
            optim_info["Param Name"].append(group["param_name"])  # 记录参数名
            optim_info["Param Shape"].append(  # 记录参数形状
                "X".join([str(x) for x in list(group["params"][0].shape)])
            )
            total_params_size += group["params"][0].numel()  # 累加参数量
            optim_info["Lr"].append(group["lr"])  # 记录学习率
            optim_info["Wd"].append(group["weight_decay"])  # 记录权重衰减
        # Counting the number of parameters
        optim_info["Param Name"].append("Total")  # 添加总计行
        optim_info["Param Shape"].append("{:.2f}M".format(total_params_size / 1e6))  # 记录总参数量（百万为单位）
        optim_info["Lr"].append("-")
        optim_info["Wd"].append("-")
        table = tabulate(  # 使用tabulate格式化优化器信息为表格
            list(zip(*optim_info.values())),
            headers=optim_info.keys(),
            tablefmt="grid",
            floatfmt=".2e",
            stralign="center",
            numalign="center",
        )
        logger = logging.getLogger("san")  # 获取名为"san"的logger
        logger.info("Optimizer Info:\n{}\n".format(table))  # 记录优化器信息
        return optimizer  # 返回构建的优化器

    @classmethod
    def test_with_TTA(cls, cfg, model):  # 定义带测试时增强（TTA）的测试方法
        logger = logging.getLogger("detectron2.trainer")  # 获取名为"detectron2.trainer"的logger
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")  # 记录开始TTA推理的信息
        model = SemanticSegmentorWithTTA(cfg, model)  # 创建带TTA的语义分割器模型
        evaluators = [  # 构建评估器列表
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")  # 为TTA结果指定不同的输出文件夹
            )
            for name in cfg.DATASETS.TEST  # 遍历测试数据集名称
        ]
        res = cls.test(cfg, model, evaluators)  # 执行测试
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})  # 在结果键名后添加"_TTA"后缀
        return res  # 返回TTA测试结果


def setup(args):  # 定义全局设置函数
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()  # 获取默认配置
    # for poly lr schedule
    add_deeplab_config(cfg)  # 添加DeepLab相关配置
    add_san_config(cfg)  # 添加SAN相关配置
    cfg.merge_from_file(args.config_file)  # 从配置文件合并配置
    cfg.merge_from_list(args.opts)  # 从命令行选项合并配置
    cfg.freeze()  # 冻结配置
    default_setup(cfg, args)  # 执行detectron2的默认设置
    if not args.eval_only:  # 如果不是仅评估模式
        setup_wandb(cfg, args)  # 设置Wandb
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="san")  # 设置名为"san"的logger
    return cfg  # 返回配置对象


def main(args):  # 定义主函数
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = setup(args)  # 执行设置

    if args.eval_only:  # 如果是仅评估模式
        model = Trainer.build_model(cfg)  # 构建模型
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(  # 加载模型权重
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # 执行测试
        if cfg.TEST.AUG.ENABLED:  # 如果启用了测试时增强
            res.update(Trainer.test_with_TTA(cfg, model))  # 执行TTA测试并更新结果
        if comm.is_main_process():  # 如果是主进程
            verify_results(cfg, res)  # 验证结果
        return res  # 返回结果

    trainer = Trainer(cfg)  # 创建训练器对象

    trainer.resume_or_load(resume=args.resume)  # 恢复或加载训练状态
    return trainer.train()  # 开始训练并返回训练结果


if __name__ == "__main__":  # 如果作为主脚本运行
    args = default_argument_parser().parse_args()  # 解析命令行参数
    print("Command Line Args:", args)  # 打印命令行参数
    launch(  # 启动（可能是分布式）训练
        main,  # 主函数
        args.num_gpus,  # GPU数量
        num_machines=args.num_machines,  # 机器数量
        machine_rank=args.machine_rank,  # 当前机器排名
        dist_url=args.dist_url,  # 分布式通信URL
        args=(args,),  # 传递给主函数的参数
    )
