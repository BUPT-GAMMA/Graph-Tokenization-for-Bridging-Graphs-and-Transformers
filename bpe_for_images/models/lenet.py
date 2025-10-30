"""
LeNet-5分类器
=============

经典的LeNet-5 CNN架构（LeCun et al., 1998）
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5卷积神经网络
    
    经典架构:
    Input(1, 28, 28) 
    -> Conv1(6, 5x5) -> ReLU -> MaxPool(2x2)
    -> Conv2(16, 5x5) -> ReLU -> MaxPool(2x2)
    -> Flatten -> FC1(120) -> ReLU
    -> FC2(84) -> ReLU
    -> FC3(num_classes)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv1_out: int = 6,
        conv2_out: int = 16,
        fc1_out: int = 120,
        fc2_out: int = 84
    ):
        """
        Args:
            in_channels: 输入通道数（灰度图为1）
            num_classes: 分类类别数
            conv1_out: 第一个卷积层输出通道数
            conv2_out: 第二个卷积层输出通道数
            fc1_out: 第一个全连接层输出维度
            fc2_out: 第二个全连接层输出维度
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度
        # MNIST: 28x28 -> conv1+pool1 -> 14x14 -> conv2+pool2 -> 5x5
        self.flatten_size = conv2_out * 5 * 5
        
        # 全连接层
        self.fc1 = nn.Linear(self.flatten_size, fc1_out)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(fc2_out, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming初始化（适合ReLU）"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, 1, 28, 28) 或 (batch_size, 784)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # 如果输入是展平格式，恢复为图像格式
        if x.dim() == 2:  # (batch, 784)
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:  # (batch, 28, 28)
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 卷积层
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self) -> int:
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import LENET_CONFIG
    
    print("测试LeNet-5分类器...")
    
    # 创建模型
    model = LeNet5(**LENET_CONFIG)
    print(f"\n模型架构:")
    print(model)
    print(f"\n参数数量: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 32
    
    # 测试图像输入
    x_image = torch.randn(batch_size, 1, 28, 28)
    out1 = model(x_image)
    print(f"\n图像输入测试: {x_image.shape} -> {out1.shape}")
    
    # 测试展平输入
    x_flatten = torch.randn(batch_size, 784)
    out2 = model(x_flatten)
    print(f"展平输入测试: {x_flatten.shape} -> {out2.shape}")
    
    # 测试无通道维度输入
    x_no_channel = torch.randn(batch_size, 28, 28)
    out3 = model(x_no_channel)
    print(f"无通道输入测试: {x_no_channel.shape} -> {out3.shape}")
    
    print("\n测试通过！")

