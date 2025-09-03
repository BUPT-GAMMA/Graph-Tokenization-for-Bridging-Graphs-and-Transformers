#!/bin/bash

# =============================================================================
# TokenizerGraph 微调包装脚本
# =============================================================================
#
# 功能说明：
#   无论从哪里调用，都会进入项目目录，然后运行微调脚本
#   保持所有传入的参数完整传递
#
# 使用方法：
#   ./finetune_wrapper.sh --dataset qm9 --method feuler [其他参数...]
#   或者从任何目录调用：
#   /path/to/project/finetune_wrapper.sh --dataset qm9 --method feuler [其他参数...]
#
# =============================================================================

# 项目根目录（写死路径，确保ClearML Agent也能正确工作）
PROJECT_ROOT="/home/gzy/py/tokenizerGraph"

# 切换到项目目录
cd "$PROJECT_ROOT" || {
    echo "❌ 错误：无法切换到项目目录 $PROJECT_ROOT"
    exit 1
}
cd "$PROJECT_ROOT"
echo "📂 当前工作目录: $(pwd)"
echo "🎯 项目根目录: $PROJECT_ROOT"
echo "🚀 启动微调脚本..."


# 检查是否通过ClearML传递了完整的命令
if [ $# -eq 2 ] && [ "$1" = "--command" ]; then
    # 通过ClearML传递的完整命令，解析并执行
    COMMAND="$2"
    echo "🔄 执行ClearML命令: $COMMAND"

    # 解析命令并执行
    eval "$COMMAND"
else
    # 正常参数传递
    exec python run_finetune.py "$@"
fi
