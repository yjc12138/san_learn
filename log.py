import os
import re

# 定义数据集映射到更友好的名称
dataset_name_map = {
    'coco_2017_test_stuff_sem_seg': 'COCO',
    'voc_sem_seg_val': 'VOC',
    'pcontext_sem_seg_val': 'PC-59',
    'ade20k_sem_seg_val': 'ADE-150',
    'pcontext_full_sem_seg_val': 'PC-459',
    'ade20k_full_sem_seg_val': 'ADE-847'
}

# 定义显示顺序
display_order = ['ADE-847', 'PC-459', 'ADE-150', 'PC-59', 'VOC', 'COCO']

def extract_miou_from_log(log_file):
    """从日志文件中提取每个数据集的最后一次mIoU结果"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 针对每个数据集提取结果
    for dataset in dataset_name_map.keys():
        # 匹配格式: Evaluation results for dataset_name in csv format:
        pattern = rf'Evaluation results for {dataset}.*?copypaste: (\d+\.\d+),.*?'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            # 取最后一次匹配的结果
            miou = float(matches[-1])
            friendly_name = dataset_name_map.get(dataset, dataset)
            results[friendly_name] = miou
    
    return results

def print_results(results):
    """以指定顺序打印结果"""
    if not results:
        print("没有找到有效的评估结果")
        return
    
    # 打印表头
    print("\n提取的评估结果 (mIoU):")
    header = "Dataset" + " " * 8 + " | mIoU"
    print(header)
    print("-" * len(header))
    
    # 按指定顺序打印结果
    for dataset in display_order:
        if dataset in results:
            miou = results[dataset]
            print(f"{dataset:<15} | {miou:>5.2f}")
        else:
            print(f"{dataset:<15} | N/A")

def main():
    # 使用固定的日志文件路径
    log_file = "output/train_vit_14_3_test/log.txt"
    log_file = "output/train_vit_14_2/log.txt"
    
    if not os.path.exists(log_file):
        print(f"错误: 文件 '{log_file}' 不存在")
        return
    
    print(f"正在分析日志文件: {log_file}")
    results = extract_miou_from_log(log_file)
    print_results(results)

if __name__ == "__main__":
    main()