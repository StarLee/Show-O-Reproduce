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