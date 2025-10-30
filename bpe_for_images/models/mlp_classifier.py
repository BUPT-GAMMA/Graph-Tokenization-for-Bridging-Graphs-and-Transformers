"""
MLP分类器
=========

简单的多层感知机baseline
"""

import torch
import torch.nn as nn
from typing import List


class MLPClassifier(nn.Module):
    """
    多层感知机分类器
    
    架构: Input -> Hidden Layers -> Output
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [512, 256],
        num_classes: int = 10,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        """
        Args:
            input_size: 输入维度（MNIST展平后为784）
            hidden_sizes: 隐藏层尺寸列表
            num_classes: 分类类别数
            dropout: Dropout比例
            activation: 激活函数类型（"relu", "gelu", "tanh"）
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # 选择激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, input_size) 或 (batch_size, 1, 28, 28)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # 如果输入是图像格式，展平
        if x.dim() == 4:  # (batch, 1, 28, 28)
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:  # (batch, 28, 28)
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def count_parameters(self) -> int:
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MLP_CONFIG
    
    print("测试MLP分类器...")
    
    # 创建模型
    model = MLPClassifier(**MLP_CONFIG)
    print(f"\n模型架构:")
    print(model)
    print(f"\n参数数量: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 32
    
    # 测试展平输入
    x_flatten = torch.randn(batch_size, 784)
    out1 = model(x_flatten)
    print(f"\n展平输入测试: {x_flatten.shape} -> {out1.shape}")
    
    # 测试图像输入
    x_image = torch.randn(batch_size, 1, 28, 28)
    out2 = model(x_image)
    print(f"图像输入测试: {x_image.shape} -> {out2.shape}")
    
    print("\n测试通过！")

