import pickle
import sys
import pprint

# 将项目根目录添加到Python路径中，这对于pickle正确加载
# 在项目中定义的自定义类的实例非常重要。
sys.path.append('/home/gzy/py/tokenizerGraph')

file_path = 'model/bpe/zinc/smiles/multi_100/bpe_codebook.pkl'

try:
    with open(file_path, 'rb') as f:
        # 使用pickle加载数据
        data = pickle.load(f)

    print(f"✅ 成功加载文件: {file_path}")
    print("="*50)
    
    # 打印加载数据的类型
    print(f"数据类型: {type(data)}")
    print("="*50)

    # 如果是字典，打印其键和一些示例内容
    if isinstance(data, dict):
        print("字典的键 (Keys):")
        pprint.pprint(list(data.keys()))
        print("-" * 50)
        
        for key, value in data.items():
            if key == 'merge_rules':
                print(f"🔍 检查 'merge_rules':")
                print(f"  - 类型: {type(value)}")
                if isinstance(value, list):
                    print(f"  - 规则数量: {len(value)}")
                    print(f"  - 前5条规则: ")
                    pprint.pprint(value[:5])
                    print(f"  - 后5条规则: ")
                    pprint.pprint(value[-5:])
            else:
                print(f"🔑 Key: {key}")
                print(f"  - 类型: {type(value)}")
                print(f"  - 内容: {value}")
    else:
        # 如果不是字典，直接打印数据
        print("数据内容:")
        pprint.pprint(data)

except FileNotFoundError:
    print(f"❌ 错误: 文件未找到 {file_path}")
except Exception as e:
    print(f"❌ 加载或检查文件时发生错误: {e}")
