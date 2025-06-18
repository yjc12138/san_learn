import os
from pathlib import Path

# 配置路径
data_root = Path("/home/Tarkiya/project/NLP/code/yjc/data")
pcontext_dir = data_root / "pcontext"
pcontext_full_dir = data_root / "pcontext_full"
image_source_dir = pcontext_dir / "JPEGImages"  # 假设图像在这个目录
gt_source_dir = pcontext_dir / "annotations_detectron2" / "pc459_val"

# 创建必要的目录结构
def create_directories():
    os.makedirs(pcontext_full_dir, exist_ok=True)
    os.makedirs(pcontext_full_dir / "val", exist_ok=True)
    print(f"创建目录: {pcontext_full_dir}")
    print(f"创建目录: {pcontext_full_dir / 'val'}")

# 创建软链接
def create_symlinks():
    # 创建图像目录的软链接
    image_link_path = pcontext_full_dir / "val" / "image"
    if os.path.exists(image_link_path):
        if os.path.islink(image_link_path):
            os.unlink(image_link_path)
        else:
            print(f"警告: {image_link_path} 已存在且不是软链接，无法创建软链接")
            return
    
    os.symlink(image_source_dir, image_link_path)
    print(f"创建软链接: {image_link_path} -> {image_source_dir}")
    
    # 创建标注目录的软链接
    label_link_path = pcontext_full_dir / "val" / "label"
    if os.path.exists(label_link_path):
        if os.path.islink(label_link_path):
            os.unlink(label_link_path)
        else:
            print(f"警告: {label_link_path} 已存在且不是软链接，无法创建软链接")
            return
    
    os.symlink(gt_source_dir, label_link_path)
    print(f"创建软链接: {label_link_path} -> {gt_source_dir}")

def main():
    print("开始设置 pcontext_full 数据集的软链接...")
    
    # 检查源目录是否存在
    if not image_source_dir.exists():
        print(f"错误: 图像源目录 {image_source_dir} 不存在")
        return
    
    if not gt_source_dir.exists():
        print(f"错误: 标注源目录 {gt_source_dir} 不存在")
        return
    
    # 创建目录结构
    create_directories()
    
    # 创建软链接
    create_symlinks()
    
    print("完成! pcontext_full 数据集的软链接已设置")

if __name__ == "__main__":
    main() 