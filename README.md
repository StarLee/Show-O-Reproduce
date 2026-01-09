

# 0，提前说明
## 0.1 项目说明
+ 复现 Show-O 统一多模型模型的训练与推理过程
+ 期待输出：
    - 整体训练过程跑通
    - 进行 图片理解，图片生成，图片补全，图片扩展推理。

## 0.2 环境准备
+ Linux/4GPU (t4 16G)
+ python 3.11.x
+ pip install -r  requirements.txt  安装依赖
+ 准备好 magvitv2 与 phi-1.5 模型权重
+ 准备好 wandb 账号（挂梯子），并在本机登陆

# 1, Train 复现
## 1.1 数据集准备
### 1.1.0 提前准备
+ 先提前准备好 hf 相关环境，推荐使用 [https://hf-mirror.com/](https://hf-mirror.com/)，见其中的 **方法三**。
+ 准备 hf 相关 token（imagenet-1k，falcon-refinedweb 需要）
+ 因为数据集都很大，建议下载完部分后，就先暂停，用局部样本先跑通整个流程
+ 训练有三个阶段，模型结构一样，只是数据集不一样，以下只针对 stage1 进行说明；另外两个 stage 更换对应数据集即可。有多个数据集，总体集合很大，全量训练时，建议通过 S3 挂盘的形式，而不是单机多卡，在单机上存放数据集。建议多机多卡。

### 1.1.1，fefinedWeb 数据集
通过 hf 下载，

```python
export HF_TOKEN="your token";hfd.sh tiiuae/falcon-refinedweb --dataset
```

**注意**： 要重写 原作者实现的分布式读取 DataLoader 的实现，代码在parquet/refinedweb_dataset.py，因为CruiseParquetDataset 是作者内部的类。改写如下

```python
import torch
import random
import glob
import os
import collections
import pandas as pd # 依然需要，用于将 Arrow Table 转为 DataFrame
import pyarrow.parquet as pq # 新增依赖
from torch.utils.data import IterableDataset, get_worker_info

class RefinedWebDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1,
                 **kwargs
                ):
        super().__init__()
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.max_length = max_length

        # 1. 扫描所有 Parquet 文件
        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        elif os.path.isfile(data_path):
            self.files = [data_path]
        else:
            self.files = sorted(glob.glob(data_path))

        if not self.files:
            raise FileNotFoundError(f"No parquet files found at {data_path}")

    def _get_worker_files(self):
        """根据 rank 和 worker_id 分配文件"""
        # GPU 级分片
        gpu_files = [f for i, f in enumerate(self.files) if i % self.world_size == self.rank]

        # Worker 级分片
        worker_info = get_worker_info()
        if worker_info is None:
            my_files = gpu_files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            my_files = [f for i, f in enumerate(gpu_files) if i % num_workers == worker_id]

        return my_files

    def _data_generator(self):
        """
        核心生成器：使用 PyArrow 迭代读取 RowGroups
        """
        files = self._get_worker_files()

        while True: # Repeat loop
            if self.shuffle:
                random.shuffle(files)

            for file_path in files:
                try:
                    # 1. 打开 Parquet 文件句柄（不加载数据）

                    pf = pq.ParquetFile(file_path)

                    # 2. 获取 RowGroups 索引
                    num_row_groups = pf.num_row_groups
                    row_group_indices = list(range(num_row_groups))

                    # 3. 如果需要 shuffle，打乱 RowGroups 的读取顺序
                    # 这是实现局部随机性的关键，避免每次都按顺序读取
                    if self.shuffle:
                        random.shuffle(row_group_indices)

                    # 4. 逐个读取 RowGroup
                    for rg_index in row_group_indices:
                        # 读取单个 RowGroup 到内存
                        table = pf.read_row_group(rg_index)

                        # 转换为 Pandas DataFrame (内存开销仅为一个 RowGroup 的大小)
                        df = table.to_pandas()

                        # 转换为字典列表
                        records = df.to_dict('records')
                        
                        # 5. 在 RowGroup 内部进行 shuffle
                        if self.shuffle:
                            random.shuffle(records)
                            
                        for row in records:
                            yield row

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
            
            if not self.repeat:
                break

    def __iter__(self):
        iterator = self._data_generator()
        
        for data in iterator:
            try:
                # 原始的数据处理逻辑
                if 'content' not in data:
                    continue
                    
                text = data['content']
                # 简单的类型检查，防止非字符串数据
                if not isinstance(text, str):
                    continue
                
                text = text.replace('\n', '')
                
                if len(text) > self.max_length:
                    start_index = random.randint(0, len(text) - self.max_length - 1)
                    selected_text = text[start_index:start_index + self.max_length]
                else:
                    selected_text = text
                
                ret = {'input_ids': selected_text}
                yield ret

            except Exception as e:
                # print('internal dataset iter error', e)
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched
```

### 1.1.2 imagenet-1k 数据集
```python
hfd.sh ILSVRC/imagenet-1k --dataset --hf_token your_token --hf_username your username
```

**注意**：从 hf 上下载的 imagenet-1k 数据集包括 test,train,validate，且是 parquet 格式，使用时需要转换为平时使用的标准格式，转换代码如下

```python
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
```

### 1.1.3 sa1b 与 SAM-LLaVA-Captions10M 数据集(图文对) 
```python
hfd.sh ssbai/sa1b --dataset -j 1  # -j 1是我只想下载一个

hfd.sh PixArt-alpha/SAM-LLaVA-Captions10M --dataset
```

**注意**：SAM-LLaVA-Captions10M下载下来的是一个 tar.gz 文件，实际他是一个关于 sa1b 中文件对应图片的文本说明，所以需要解压，或形成一个 大的 key-value 对，但这个文件夹中有 1000W+ 文本文件，建议直接解压在 s3 上（一般 LLM 都把数据集放 S3 （OSS）上，再挂载到本机（千万不要 ls 这个目录）；或先形成一个 json 大文件（这个文件大概会有 6 个 G，加载到内存中会超过 6 个 G，单机多卡时，会极大压缩系统内存，但本次先形成 json 大文件跑过代码），转换如下

```python
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
```

同时我也提供了解压到 s3 上

```python
import tarfile
import boto3
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
from botocore.client import Config
from botocore.exceptions import ClientError

# ================= 配置区域 =================
# 1. 文件路径
LOCAL_TAR_PATH = "datasets/SAM-LLaVA-Captions10M/SA1B_caption.tar.gz"

# 2. S3 配置
S3_ENDPOINT_URL = "http://oss-ak-beijing-fxc.cecloud.com:6068"
S3_ACCESS_KEY   = "your access key"
S3_SECRET_KEY   = "your secret key"
S3_BUCKET_NAME  = "bucket-datasets"
S3_PREFIX       = "sa1b_captions/"

# 3. 性能核心调优
# 因为不能预加载，每个文件都要多一次网络请求 (Head)。
# 为了抵消这个延迟，我们需要更多的线程并发。
MAX_WORKERS     = 256  # 建议设置 64 ~ 256 之间
SKIP_EXISTING   = True  # True: 遇到已存在则跳过; False: 强制覆盖

# SSL 配置
VERIFY_SSL      = False 
# ===========================================

# 初始化 S3 客户端
# 注意：max_pool_connections 必须大于 MAX_WORKERS，否则线程会阻塞等待连接
s3_config = Config(
    s3={'addressing_style': 'auto'},
    retries={'max_attempts': 3},           # 自动重试
    max_pool_connections=MAX_WORKERS + 20, # 连接池大小 > 线程数
    connect_timeout=60,
    read_timeout=60
)

s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    use_ssl=S3_ENDPOINT_URL.startswith("https"),
    verify=VERIFY_SSL,
    config=s3_config
)

# 统计跳过数量
skipped_counter = 0
skipped_lock = threading.Lock()

def process_and_upload(bucket, key, data_bytes):
    """
    工作线程函数：检查 -> (不存在则)上传
    """
    global skipped_counter
    
    # --- 1. 检查是否存在 (Head Object) ---
    if SKIP_EXISTING:
        try:
            # Head 请求非常轻量，只返回元数据
            s3_client.head_object(Bucket=bucket, Key=key)
            # 如果没抛异常，说明文件存在 -> 跳过
            return "SKIPPED"
        except ClientError as e:
            # 如果是 404 (Not Found)，说明文件不存在，继续往下走
            error_code = e.response.get('Error', {}).get('Code')
            if error_code != "404":
                return f"检查失败 {key}: {e}"
            # 其他情况继续执行上传
    
    # --- 2. 上传文件 (Put Object) ---
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data_bytes,
            ContentType='text/plain'
        )
        return True
    except Exception as e:
        return f"上传失败 {key}: {e}"

def main():
    print(f"--- 任务开始 (实时检查模式) ---")
    print(f"并发线程数: {MAX_WORKERS}")
    
    # 线程池
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    # 信号量：防止读取 tar 的速度太快撑爆内存
    # 允许积压的任务数 = 线程数 * 5
    semaphore = threading.Semaphore(MAX_WORKERS * 5)

    def task_done_callback(future):
        """任务完成回调"""
        semaphore.release() # 释放信号量，允许主线程读下一个
        result = future.result()
        
        if result == "SKIPPED":
            with skipped_lock:
                global skipped_counter
                skipped_counter += 1
                # 实时更新跳过数 (放在 postfix 避免刷屏)
                pbar.set_postfix(skipped=skipped_counter, refresh=False)
        elif result is not True:
            # 如果是错误信息，打印出来
            tqdm.write(str(result))
        
        pbar.update(1)

    try:
        # 打开 Tar 文件流
        # mode="r|gz" (流式读取) 比 "r:gz" (随机读取) 内存占用更低，适合大文件
        with tarfile.open(LOCAL_TAR_PATH, "r|gz") as tar:
            
            global pbar
            # mininterval=1.0 防止进度条刷新太快占用 CPU
            pbar = tqdm(desc="Processing", unit=" files", mininterval=1.0, smoothing=0.1)

            for member in tar:
                # 必须是文件
                if member.isfile():
                    filename = os.path.basename(member.name) 
                    s3_key = os.path.join(S3_PREFIX, filename).replace("\\", "/").lstrip("/")

                    # 读取内容 (Tar流式读取只能读一次)
                    f = tar.extractfile(member)
                    if f:
                        file_content = f.read()

                        # 申请信号量 (如果积压任务太多，这里会阻塞)
                        semaphore.acquire()

                        # 提交到线程池
                        future = executor.submit(
                            process_and_upload, 
                            S3_BUCKET_NAME, 
                            s3_key, 
                            file_content
                        )
                        future.add_done_callback(task_done_callback)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {LOCAL_TAR_PATH}")
    except Exception as e:
        print(f"主线程异常: {e}")
    finally:
        print("\n正在等待剩余任务完成...")
        executor.shutdown(wait=True)
        pbar.close()
        print(f"\n✅ 完成。总共跳过已存在文件: {skipped_counter}")

if __name__ == "__main__":
    main()
```

## 1.2 Train
### 1.2.1 提前准备
+ 网络，因为使用 wandb 进行训练管理，需要代理，请先在终端配置代码

```python
export ALL_PROXY=socks5://10.32.122.235:10808 # ip 替换为你的IP与端口
```

```python
wandb login 
```

### 1.2.2 训练
训练启动命令如下 (以下脚本在 training目录下的 training_my.sh)

```python
#!/bin/bash
cd /home/ubuntu/codespace/llm/Show-o ; 
source /home/ubuntu/codespace/llm/Show-o/.venv/bin/activate

export DS_SKIP_CUDA_CHECK=1 # deepspeed 会检查cuda版本，设置这个环境变量
accelerate launch --config_file /home/ubuntu/codespace/llm/Show-o/accelerate_configs/4_gpus_deepspeed_zero2_my.yaml \
 --main_process_port=18888 \
/home/ubuntu/codespace/llm/Show-o/training/train.py \
 config=/home/ubuntu/codespace/llm/Show-o/configs/showo_pretraining_stage1_my.yaml
```

**注意**：因为使用 T4 显卡，本显卡比较老，对 bf16 不支持，所以在配置中混合精度改成了 fp16，建议如果显卡比较新，deepspeed 下还是保持 bf16 精度，否则 fp16 有大量的 溢出。另外：相关配置文件 accelerate_config 以及 config 下的相关配置文件，后缀有 _my 的均是流程跑通，全量训练时，请按需求修改训练脚本中的配置路径。

（以下是 wandb 截图，全量训练需要大量的 GPU 时间，本训练的次数有限，只是对应能跑通，还没有明显的 loss 下降趋势，生成也还没有明显价值）

<!-- 这是一张图片，ocr 内容为：CHARTS  7 1-60F7 STEP LOSS T2I STEP_LOSS_LM STEP LOSS MMU 5.7 20.6 6.8 20.4 6.8 20.2 6.5 6.4 20 8.4 6.2 19.8 15.5 STEP 5.8 15 25 2U 15 20 30 20 15 30 -->
![](https://cdn.nlark.com/yuque/0/2026/png/46373454/1767928465787-ad927dc0-c91f-4213-91a1-347921899327.png)

<!-- 这是一张图片，ocr 内容为：ORIGINAL IMAGES V.S.RECONSTRUCTED IMAGES V.S.PREDICTED IMAGES D TICK IN SD A FROR TH BO MASK RATIO:0.57 CAPTION:HARTEBEEST MASK RATIO:0.93 COPTION:RAIN BARREL GENERATED IMAGES WALL CLOCK LIGER -->
![](https://cdn.nlark.com/yuque/0/2026/png/46373454/1767928546765-405c3e9a-2fd6-477b-bb1e-dbc6eb2a4858.png)

# 2， 推理
## 2.1 说明
+ 模型推理的测试结果均在 results 下，分别为 mmu(图片理解)，inpainting(图片补全)，extrapolation(图片扩展),t2i(文生图)
+ 执行脚本均在 inference 下对应的 shell 脚本
+ 同时会在 wandb 上也会生成相关结果

## 2.2 mmu
```python
#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_mmu.py config=configs/showo_demo_512x512.yaml \
max_new_tokens=100 \
mmu_image_root=./mmu_validation question='Please describe this image in detail. *** Do you think the image is unusual or not?'

```

说明：输入图片在 mmu_validation 下，需要理解的 prompt 见参数 question 部分“Please describe this image in detail. *** Do you think the image is unusual or not?”

输出时，理解的内容在 results/mmu 下 对应的 txt 文件

## 2.3 t2i
```python
#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_t2i.py config=configs/showo_demo_512x512.yaml \
batch_size=1 validation_prompts_file=validation_prompts/showoprompts.txt \
guidance_scale=5 generation_timesteps=50 \
mode='t2i'
```

说明：输入 prompt 为 validation_prompts/showoprompts.txt 文件

## 2.4 inpainting 
```python
#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_t2i.py config=configs/showo_demo.yaml \
batch_size=1 \
guidance_scale=1.75 generation_timesteps=16 \
mode='inpainting' prompt='A blue sports car with sleek curves and tinted windows, parked on a bustling city street.' \
image_path=./inpainting_validation/bus.jpg inpainting_mask_path=./inpainting_validation/bus_mask.webp
```

说明：输入图片为inpainting_validation/bus.jpg，mask 部分在inpainting_validation/bus_mask.webp, prompt 为'A blue sports car with sleek curves and tinted windows, parked on a bustling city street.'

## 2.5 extrapolation
```python
#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_t2i.py config=configs/showo_demo.yaml \
batch_size=1 \
guidance_scale=1.75 generation_timesteps=16 \
mode='extrapolation' extra_direction='left *** left *** left *** right *** right *** right' offset=0 prompt='a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees.' \
image_path=./inpainting_validation/alpine_lake.jpg
```

说明：输入图片为inpainting_validation/alpine_lake.jpg，扩展要求在 执行脚本的 extra_direction 部分。即

```plain
left *** left *** left *** right *** right *** right' offset=0 prompt='a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees.
```

