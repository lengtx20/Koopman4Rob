import numpy as np
import torch
from data.mcap_data_utils import BatchProcessor, BatchStackerConfig
from pprint import pprint
import timeit


# ===========================================================
# 构造测试数据
# ===========================================================
def make_sample(i):
    """构造单个样本，key 都有一个 (D,) 的向量"""
    return {
        "0.0key1": {"data": np.ones(3) * (i + 0.0)},
        "0.0key2": {"data": np.ones(2) * (i + 0.1)},
        "1.0key1": {"data": np.ones(3) * (i + 1.0)},
        "1.0key2": {"data": np.ones(2) * (i + 1.1)},
        "1.2key1": {"data": np.ones(3) * (i + 1.2)},
        "1.2key2": {"data": np.ones(2) * (i + 1.3)},
        "0.1key1": {"data": np.ones(3) * (i + 0.1)},
        "0.1key2": {"data": np.ones(2) * (i + 0.2)},
        "1.5key1": {"data": np.ones(3) * (i + 1.5)},
        "1.5key2": {"data": np.ones(2) * (i + 1.6)},
        "meta": {"data": f"sample-{i}"},
    }


# 三个样本
batched_samples = [make_sample(i) for i in range(3)]

# ===========================================================
# stack 配置
# ===========================================================
stack_config = {
    "cur_state": ["0.0key1", "0.0key2"],  # flat
    "next_action": [["key1", "key2"], [1.0, 1.2]],  # range风格
    "complex": [  # complex风格
        ["0.0key1", "0.0key2"],
        ["0.1key1", "0.1key2"],
        ["1.0key1", "1.0key2"],
        ["1.5key1", "1.5key2"],
    ],
}

# ===========================================================
# 运行测试
# ===========================================================
processor = BatchProcessor(BatchStackerConfig(stack=stack_config))
pprint(processor.stack)

batched = processor(batched_samples)
# ===========================================================
cost = timeit.timeit(lambda: processor(batched_samples), number=1000)
print(f"Total time for 1000 runs: {cost:.6f} seconds")
print(f"Average time per run: {cost:.6f} ms")
# ===========================================================
# 打印结果形状
# ===========================================================
print("=== 输出结果 ===")
for k, v in batched.items():
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(f"{k:12s} -> shape {v.shape}")
    else:
        print(f"{k:12s} -> {v}")
