

import torch
import torch.nn as nn
import torch.optim as optim
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import gc

# 初始化GPU内存监控
def get_gpu_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024 // 1024  # 返回MB

# 创建一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 测试函数
def test_memory_usage(freeze=False):
    # 清空缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 创建模型并移动到GPU
    model = SimpleModel().cuda()
    
    # 冻结部分参数
    if freeze:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False
    
    # 创建优化器（只优化需要梯度的参数）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # 记录初始内存使用
    initial_memory = get_gpu_memory()
    
    # 前向传播
    inputs = torch.randn(6400, 1000).cuda()  # batch_size=64
    outputs = model(inputs)
    
    # 后向传播
    loss = outputs.sum()
    loss.backward()
    optimizer.step()
    
    # 记录峰值内存使用
    peak_memory = get_gpu_memory()
    
    # 清理
    del model, optimizer, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return initial_memory, peak_memory

# 运行测试
print("测试不冻结参数的情况：")
initial_no_freeze, peak_no_freeze = test_memory_usage(freeze=False)
print(f"初始显存: {initial_no_freeze}MB, 峰值显存: {peak_no_freeze}MB")

print("\n测试冻结部分参数的情况：")
initial_freeze, peak_freeze = test_memory_usage(freeze=True)
print(f"初始显存: {initial_freeze}MB, 峰值显存: {peak_freeze}MB")

print(f"\n显存节省: {peak_no_freeze - peak_freeze}MB")