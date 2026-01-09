import os
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
'''
这个是用来将从hg上下载的imagenet-1k数据集（parque格式）转为标准的格式
'''
# ================= 配置区域 =================
# 你的 parquet 文件所在的文件夹路径
INPUT_DIR = "datasets/imagenet-1k-samples/data" 

# 你希望图片解压到哪去
OUTPUT_DIR = "datasets/imagenet-1k-samples/train"
# ===========================================

def save_image(image_data, label, filename, output_root):
    """
    将二进制图片数据保存为文件
    """
    # 1. 确定类别文件夹名称
    # 如果 label 是 -1 或 None (通常是 Test 集)，放入 'unknown' 文件夹
    if label is None or label == -1:
        class_dir = os.path.join(output_root, "unknown")
    else:
        # 这里直接用数字作为类别名，例如 class_0, class_1
        # 如果parquet里包含synset ID (如 n01440764)，也可以改用那个
        class_dir = os.path.join(output_root, f"class_{label}")

    os.makedirs(class_dir, exist_ok=True)

    # 2. 确定保存路径
    save_path = os.path.join(class_dir, filename)

    # 3. 解码并保存图片
    try:
        # HuggingFace 的 image 列通常是字典: {'bytes': b'...', 'path': '...'}
        # 或者直接是 bytes
        if isinstance(image_data, dict) and 'bytes' in image_data:
            img_bytes = image_data['bytes']
        elif isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            print(f"Skipping {filename}: Unknown image format")
            return

        image = Image.open(io.BytesIO(img_bytes))
        
        # 强制转换为 RGB (防止部分灰度图或CMYK图导致报错)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        image.save(save_path, "JPEG")
        
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def main():
    # 找到所有 parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    
    if not parquet_files:
        print(f"错误: 在 {INPUT_DIR} 没找到任何 .parquet 文件")
        return

    print(f"找到 {len(parquet_files)} 个 Parquet 文件，准备开始转换...")
    print(f"输出目录: {OUTPUT_DIR}")

    total_images = 0
    
    # 遍历处理每个 parquet 文件
    for p_file in parquet_files:
        print(f"正在处理: {os.path.basename(p_file)}")
        
        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(p_file)
            
            # 使用 tqdm 显示进度条
            for index, row in tqdm(df.iterrows(), total=len(df), unit="img"):
                # 尝试获取 image 列
                if 'image' not in row:
                    continue
                
                # 尝试获取 label 列 (Test 集可能没有 label)
                label = row['label'] if 'label' in row else -1
                
                # 生成一个文件名 (使用索引或原来的文件名)
                # 如果数据里有 filename 列最好，没有就自造一个
                filename = f"{os.path.basename(p_file).replace('.parquet', '')}_{index}.jpg"
                
                save_image(row['image'], label, filename, OUTPUT_DIR)
                total_images += 1
                
        except Exception as e:
            print(f"读取文件 {p_file} 失败: {e}")

    print(f"\n全部完成！共提取 {total_images} 张图片。")
    print(f"请检查目录结构是否符合: {OUTPUT_DIR}/class_x/image.jpg")

if __name__ == "__main__":
    main()