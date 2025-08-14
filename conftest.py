# conftest.py - pytest配置文件
# 在pytest启动的最早阶段设置警告过滤

import warnings
import os

# 在导入阶段就设置警告过滤，确保在所有pytest插件加载之前生效
warnings.filterwarnings('ignore', category=Warning, module='outdated')
warnings.filterwarnings('ignore', message='.*Failed to check for latest version.*')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

# 也可以通过环境变量的方式禁用outdated包的警告检查
os.environ.setdefault('OUTDATED_IGNORE', '1')

def pytest_configure(config):
    """pytest配置钩子 - 在所有配置完成后执行"""
    # 确保警告过滤规则已经设置
    warnings.filterwarnings('ignore', category=Warning, module='outdated')
    warnings.filterwarnings('ignore', message='.*Failed to check for latest version.*')
    warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')


