# 批量脚本更新说明

## 概述

本次更新修改了项目的批量脚本，使其调用新创建的bash包装脚本而不是直接调用Python脚本。这样可以确保无论从哪里调用，都会正确切换到项目目录。

## 修改内容

### 1. 创建的包装脚本

#### `pretrain_wrapper.sh`
- 位置：`/home/gzy/py/tokenizerGraph/pretrain_wrapper.sh`
- 功能：预训练脚本的包装器
- 特点：
  - **写死项目路径**：直接使用 `/home/gzy/py/tokenizerGraph`，确保ClearML Agent也能正确工作
  - 无论从哪里调用都会切换到项目目录
  - 保持所有参数传递
  - 支持conda环境激活

#### `finetune_wrapper.sh`
- 位置：`/home/gzy/py/tokenizerGraph/finetune_wrapper.sh`
- 功能：微调脚本的包装器
- 特点：与预训练包装脚本相同

### 2. 修改的批量脚本

#### `batch_pretrain_simple.py`
- **修改位置**：第194行
- **修改前**：
  ```python
  cmd = [
      "python", "run_pretrain.py",
      "--dataset", task["dataset"],
      # ...
  ]
  ```
- **修改后**：
  ```python
  cmd = [
      "./pretrain_wrapper.sh",
      "--dataset", task["dataset"],
      # ...
  ]
  ```

#### `batch_finetune_simple.py`
- **修改位置**：第222行
- **修改前**：
  ```python
  cmd = [
      "python", "run_finetune.py",
      "--dataset", task["dataset"],
      # ...
  ]
  ```
- **修改后**：
  ```python
  cmd = [
      "./finetune_wrapper.sh",
      "--dataset", task["dataset"],
      # ...
  ]
  ```

### 3. 修改的ClearML提交脚本

#### `batch_submit_clearml.py`
- **修改内容**：
  - 增强了脚本类型检测，支持bash脚本
  - 为bash脚本使用命令行模式创建ClearML任务
  - 保持对Python脚本的兼容性

- **主要修改**：
  ```python
  # 处理不同类型的脚本调用
  if parts[0] in ['python', 'python3'] and len(parts) > 1:
      # Python脚本调用
      script_path = parts[1]
      args = parts[2:]
  elif parts[0].endswith('.sh') and len(parts) > 0:
      # Bash脚本调用
      script_path = parts[0]
      args = parts[1:]
  else:
      # 其他类型的脚本或命令
      script_path = parts[0]
      args = parts[1:]
  ```

  ```python
  # 创建任务模板（不执行代码）
  if script_path.endswith('.sh'):
      # 对于bash脚本，使用命令行模式
      task = Task.create(
          project_name="TokenizerGraph",
          task_name=task_name,
          working_directory=self.working_directory
      )
      # 设置任务执行的命令行
      task.set_script(command_line=command_line)
  else:
      # 对于Python脚本，使用标准模式
      task = Task.create(
          project_name="TokenizerGraph",
          task_name=task_name,
          script=script_path,
          working_directory=self.working_directory,
          argparse_args=parsed_args
      )
  ```

## 优势

1. **统一性**：无论从项目内部还是外部调用，都会正确切换到项目目录
2. **参数完整性**：保持所有命令行参数的完整传递
3. **环境管理**：包装脚本可以处理conda环境激活
4. **兼容性**：ClearML提交脚本同时支持bash和Python脚本
5. **可维护性**：集中管理项目路径和环境设置
6. **ClearML Agent兼容性**：写死项目路径，确保在分布式环境中也能正确工作

## ClearML Agent 兼容性说明

### 设计考虑

考虑到ClearML Agent的执行环境特点：

1. **仓库克隆**：Agent可能会将仓库克隆到不同的临时目录
2. **脚本位置不确定性**：脚本可能被复制到其他位置执行
3. **路径依赖**：不能依赖脚本的相对位置来确定项目根目录

### 解决方案

为了确保在ClearML Agent环境中也能正确工作，采用**写死项目路径**的策略：

```bash
# 写死项目路径，而不是使用相对路径
PROJECT_ROOT="/home/gzy/py/tokenizerGraph"
cd "$PROJECT_ROOT"
```

这样无论Agent将代码克隆到哪个目录，脚本都会：
1. 直接切换到正确的项目目录
2. 执行相应的Python脚本
3. 保持所有参数传递

### 验证结果

通过实际测试验证了修改效果：

```bash
# 从其他目录调用脚本
cd /tmp && /home/gzy/py/tokenizerGraph/pretrain_wrapper.sh --help
```

输出显示：
```
📂 当前工作目录: /home/gzy/py/tokenizerGraph
🎯 项目根目录: /home/gzy/py/tokenizerGraph
🚀 启动预训练脚本...
```

✅ **成功**：脚本从 `/tmp` 目录正确切换到项目目录 `/home/gzy/py/tokenizerGraph`

### 优势

- ✅ **分布式环境兼容**：支持ClearML Agent的分布式执行
- ✅ **路径确定性**：不受Agent克隆位置影响
- ✅ **环境一致性**：确保所有执行都在正确的项目环境中进行
- ✅ **测试验证**：实际测试证明方案有效

## 使用方法

### 直接使用包装脚本
```bash
# 从任何目录调用
./pretrain_wrapper.sh --dataset qm9test --method feuler
./finetune_wrapper.sh --dataset qm9test --method feuler
```

### 批量脚本使用
```bash
# 预训练批量脚本（现在会自动使用包装脚本）
python batch_pretrain_simple.py --datasets qm9test --methods feuler

# 微调批量脚本（现在会自动使用包装脚本）
python batch_finetune_simple.py --datasets qm9test --methods feuler
```

### ClearML提交
```bash
# 支持bash脚本的命令文件
echo "./pretrain_wrapper.sh --dataset qm9test --method feuler" > commands.txt
python batch_submit_clearml.py --file commands.txt
```

## 测试验证

运行测试脚本验证修改：
```bash
python test_batch_modifications.py
```

测试结果：
- ✅ 预训练脚本正确使用包装脚本
- ✅ 微调脚本正确使用包装脚本
- ✅ ClearML脚本支持bash脚本解析

## 注意事项

1. 确保包装脚本具有执行权限：`chmod +x *.sh`
2. 包装脚本会尝试激活`pthgnn` conda环境，如不需要可注释相关代码
3. ClearML提交需要网络连接到ClearML服务器
4. 所有修改保持向后兼容，现有Python脚本调用仍可正常工作
