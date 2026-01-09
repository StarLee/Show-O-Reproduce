import tarfile
import json
import os
import io
from tqdm import tqdm

# ================= 配置区域 =================
# 输入的压缩包路径
TAR_PATH = "datasets/SAM-LLaVA-Captions10M/SA1B_caption.tar.gz" 

# 输出的 JSON 文件路径
OUTPUT_PATH = "datasets/SAM-LLaVA-Captions10M/sa1b_captions.json"

# 如果你的文件都在一个子目录下（比如 all/sa_00000.txt），是否要去掉目录前缀？
# True: 键为 "sa_00000"
# False: 键为 "all/sa_00000" (通常 Show-o 代码逻辑是用 split('/')[-1]，所以 True 比较安全)
STRIP_PATH = True
# ===========================================

def convert_tar_to_json():
    print(f"正在读取: {TAR_PATH} ...")
    print("注意：这可能需要几分钟，具体取决于文件大小。")

    # 用于存储结果的大字典
    data_dict = {}
    
    # 计数器
    count = 0

    try:
        # 使用 'r:gz' 模式打开，以流的方式读取
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            # 获取迭代器（这样不会一次性加载所有元数据到内存）
            member = tar.next()
            
            # 使用 tqdm 显示进度（由于流式读取难以预知总数，这里只显示已处理数量）
            pbar = tqdm(unit=" files", desc="Processing")
            
            while member:
                # 只处理文件，跳过文件夹
                if member.isfile():
                    try:
                        # 1. 提取文件名作为 Key
                        filename = member.name
                        if STRIP_PATH:
                            # 去掉路径，只留文件名 (例如 all/abc.txt -> abc.txt)
                            filename = os.path.basename(filename)
                        
                        # 去掉扩展名 (例如 abc.txt -> abc)
                        # 这一步很重要，因为你的代码里是用 key 去匹配的
                        key = os.path.splitext(filename)[0]

                        # 2. 读取文件内容作为 Value
                        f = tar.extractfile(member)
                        if f:
                            # 假设是 utf-8 编码的文本
                            content = f.read().decode('utf-8').strip()
                            
                            # 存入字典
                            data_dict[key] = content
                            
                            count += 1
                            if count % 1000 == 0:
                                pbar.update(1000)
                                
                    except Exception as e:
                        print(f"\n跳过文件 {member.name}: {e}")

                # 读取下一个成员
                member = tar.next()
            
            pbar.close()

    except FileNotFoundError:
        print(f"错误: 找不到文件 {TAR_PATH}")
        return
    except Exception as e:
        print(f"发生错误: {e}")
        return

    print(f"\n读取完成，共提取 {len(data_dict)} 条数据。")
    print(f"正在写入 JSON 文件: {OUTPUT_PATH} ...")

    # 3. 将大字典写入 JSON 文件
    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as json_file:
            # ensure_ascii=False 可以让中文正常显示（如果有的话）
            # indent=None 不换行，减小文件体积
            json.dump(data_dict, json_file, ensure_ascii=False)
        print("写入成功！")
        
    except MemoryError:
        print("错误：内存不足，无法一次性写入 JSON。建议改用 JSONL 格式或增加机器内存。")

if __name__ == "__main__":
    convert_tar_to_json()