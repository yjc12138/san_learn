import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 配置路径
data_root = Path("/home/Tarkiya/project/NLP/code/yjc/data")
pcontext_dir = data_root / "pcontext"
annotations_dir = pcontext_dir / "annotations_detectron2" / "pc59_val"
target_image_dir = pcontext_dir / "annotations_detectron2" / "pc59_val" / "image"

# 使用已有的图像目录
voc_image_dir = pcontext_dir / "JPEGImages"

# 创建目标图像目录
os.makedirs(target_image_dir, exist_ok=True)

# 获取标注文件的基本名称（不带扩展名）
def get_annotation_basenames():
    return [os.path.splitext(f)[0] for f in os.listdir(annotations_dir) if f.endswith('.png')]

# 获取所有可用的图像文件
def get_available_images():
    image_files = glob.glob(str(voc_image_dir / "*.jpg"))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}

# 检查图像文件是否有效
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图像文件
        return True
    except Exception as e:
        print(f"Invalid image file {file_path}: {e}")
        return False

# 复制对应的图像文件
def copy_images(basenames, available_images):
    print("Copying images...")
    found_count = 0
    missing_count = 0
    invalid_count = 0
    
    # 检查是否有年份不匹配的情况
    year_mismatches = {}
    for basename in basenames:
        # 尝试不同年份的匹配
        if basename.startswith("2010_") and basename not in available_images:
            alt_basename = "2008" + basename[4:]
            if alt_basename in available_images:
                year_mismatches[basename] = alt_basename
        elif basename.startswith("2008_") and basename not in available_images:
            alt_basename = "2010" + basename[4:]
            if alt_basename in available_images:
                year_mismatches[basename] = alt_basename
    
    if year_mismatches:
        print(f"Found {len(year_mismatches)} files with year mismatches in the filename.")
        print("Examples:")
        for i, (orig, alt) in enumerate(list(year_mismatches.items())[:5]):
            print(f"  {orig} -> {alt}")
        
        use_alt = input("Do you want to use these alternative filenames? (y/n): ").lower() == 'y'
    else:
        use_alt = False
    
    for basename in tqdm(basenames):
        # 如果有年份不匹配且用户同意使用替代文件名
        if use_alt and basename in year_mismatches:
            src_basename = year_mismatches[basename]
            src_path = Path(available_images[src_basename])
        else:
            src_basename = basename
            if src_basename in available_images:
                src_path = Path(available_images[src_basename])
            else:
                print(f"Warning: Image for {basename} not found.")
                missing_count += 1
                continue
        
        dst_path = target_image_dir / f"{basename}.jpg"
        
        # 验证源图像是否有效
        if is_valid_image(src_path):
            shutil.copy2(src_path, dst_path)
            
            # 验证复制后的图像是否有效
            if is_valid_image(dst_path):
                found_count += 1
            else:
                print(f"Warning: Copied image {dst_path} is invalid.")
                invalid_count += 1
                # 尝试删除无效文件
                try:
                    os.remove(dst_path)
                except:
                    pass
        else:
            print(f"Warning: Source image {src_path} is invalid.")
            invalid_count += 1
    
    print(f"Copied {found_count} images. {missing_count} images were missing. {invalid_count} images were invalid.")

# 创建符号链接（可选，如果您想保持原始目录结构）
def create_symlink():
    if not os.path.exists(pcontext_dir / "val"):
        os.makedirs(pcontext_dir / "val", exist_ok=True)
    
    if not os.path.exists(pcontext_dir / "val" / "image"):
        os.symlink(target_image_dir, pcontext_dir / "val" / "image")
    
    if not os.path.exists(pcontext_dir / "val" / "label"):
        os.symlink(annotations_dir, pcontext_dir / "val" / "label")

# 主函数
def main():
    # 检查图像目录是否存在
    if not voc_image_dir.exists():
        print(f"Error: Image directory {voc_image_dir} does not exist.")
        return
    
    print(f"Using image directory: {voc_image_dir}")
    
    # 获取标注文件的基本名称
    basenames = get_annotation_basenames()
    print(f"Found {len(basenames)} annotation files.")
    
    # 获取所有可用的图像文件
    available_images = get_available_images()
    print(f"Found {len(available_images)} available image files.")
    
    # 检查匹配情况
    matched = [b for b in basenames if b in available_images]
    print(f"Direct filename matches: {len(matched)} out of {len(basenames)}")
    
    # 复制对应的图像文件
    copy_images(basenames, available_images)
    
    # 创建符号链接（可选）
    create_symlink()
    
    print("Done! Images are now set up correctly.")
    print(f"Images are located at: {target_image_dir}")
    print(f"Annotations are located at: {annotations_dir}")

if __name__ == "__main__":
    main() 