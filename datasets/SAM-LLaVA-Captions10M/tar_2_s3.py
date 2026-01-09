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
S3_ENDPOINT_URL = "xxx"
S3_ACCESS_KEY   = "xxx"
S3_SECRET_KEY   = "xxx"
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