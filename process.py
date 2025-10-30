import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from module import save, load, makedir_exist_ok
from collections import defaultdict
import re

result_path = os.path.join('output', 'result')
save_format = 'jpg'
vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
num_experiments = 4
exp = [str(42+x) for x in list(range(num_experiments))]
dpi = 300
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'


def make_control(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_all_controls(mode):
    if mode == 'base':
        data_names = ['MNIST', 'CIFAR10', 'CIFAR100']
        # base_model_names = ['linear', 'mlp', 'cnn', 'ffn', 'cnnffn']
        base_model_names = ['resnetl18', 'resnet18', 'resnetm18']#,"cnn"]
        
        # 添加各种ffnmoe配置
        expert_numbers = [32]  # 各种专家数量
        top_k_values = [1, 2, 8,16,32]          # 各种top_k值
        ffnmoe_models = []
        
        # for e in expert_numbers:
        #     for t in top_k_values:
        #       ffnmoe_models.append(f'ffnmoe_E{e}T{t}')
        # for e in expert_numbers:
        #     for t in top_k_values:
        #       ffnmoe_models.append(f'cnnffnmoe_E{e}T{t}')
        
        # 合并所有模型名称
        model_names = base_model_names + ffnmoe_models
        control_name = [[data_names, model_names]]
        controls = make_control(control_name)
        # 删除所有mnist的t16
        controls = [control for control in controls if not ('MNIST' in control[1] and 'T16' in control[1])]
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    modes = ['base']
    controls = []
    for mode in modes:
        controls += make_all_controls(mode)
    processed_result = process_result(controls)
    df_mean = make_df(processed_result, 'mean')
    df_history = make_df(processed_result, 'history')
    make_vis_history(df_history)
    return


def tree():
    return defaultdict(tree)


def process_result(controls):
    result = tree()
    for control in controls:
        tag = '_'.join(control)
        gather_result(list(control), tag, result)
    summarize_result(None, result)
    save(result, os.path.join(result_path, 'processed_result'))
    processed_result = tree()
    extract_result(processed_result, result, [])
    return processed_result


def gather_result(control, tag, processed_result):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}'.format(tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for split in base_result['logger']:
                for metric_name in base_result['logger'][split]['mean']:
                    processed_result[split][metric_name]['mean'][exp_idx] \
                        = base_result['logger'][split]['mean'][metric_name]
                for metric_name in base_result['logger'][split]['history']:
                    processed_result[split][metric_name]['history'][exp_idx] \
                        = base_result['logger'][split]['history'][metric_name]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        gather_result([control[0]] + control[2:], tag, processed_result[control[1]])
    return


def summarize_result(key, value):
    if key in ['mean', 'history']:
        value['summary']['value'] = np.stack(list(value.values()), axis=0)
        value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
        value['summary']['std'] = np.std(value['summary']['value'], axis=0)
        value['summary']['max'] = np.max(value['summary']['value'], axis=0)
        value['summary']['min'] = np.min(value['summary']['value'], axis=0)
        value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
        value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
        value['summary']['value'] = value['summary']['value'].tolist()
    else:
        for k, v in value.items():
            summarize_result(k, v)
        return
    return


def extract_result(extracted_processed_result, processed_result, control):
    def extract(split, metric_name, mode):
        output = False
        if split == 'train':
            if metric_name in ['test/Loss', 'test/Accuracy']:
                if mode == 'history':
                    output = True
        elif split == 'test':
            if metric_name in ['test/Loss', 'test/Accuracy']:
                if mode == 'mean':
                    output = True
        return output

    if 'summary' in processed_result:
        control_name, split, metric_name, mode = control
        if not extract(split, metric_name, mode):
            return
        stats = ['mean', 'std']
        for stat in stats:
            exp_name = '_'.join([control_name, metric_name.split('/')[1], stat])
            extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            extract_result(extracted_processed_result, v, control + [k])
    return


def make_df(processed_result, mode):
    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        exp_name_list = exp_name.split('_')
        df_name = '_'.join([*exp_name_list])
        index_name = [1]
        df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
    startrow = 0
    with pd.ExcelWriter(os.path.join(result_path, 'result_{}.xlsx'.format(mode)), engine='xlsxwriter') as writer:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1, header=False, index=False)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
    return df


def make_vis_history(df_history):
    # 扩展字典以支持ffnmoe模型
    label_dict = {
        'linear': 'Linear', 
        'mlp': 'MLP', 
        'cnn': 'CNN', 
        'resnet18': 'ResNet18',
        'ffn': 'FFN'
    }
    
    # 基础颜色字典
    color_dict = {
        'linear': 'red', 
        'mlp': 'orange', 
        'cnn': 'blue', 
        'resnet18': 'dodgerblue',
        'ffn': 'green',
        'resnetm18': 'purple',
        'resnetl18': 'gold'
    }
    
    # 基础线型字典
    linestyle_dict = {
        'linear': '-', 
        'mlp': '--', 
        'cnn': ':', 
        'resnet18': '-.',
        'ffn': '-',
        'resnetm18': '-',
        'resnetl18': '-'
    }
    
    # 颜色列表用于ffnmoe模型
    moe_colors = ['purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'coral', 'gold']
    
    # 线型列表用于ffnmoe模型
    moe_linestyles = ['-', '--', ':', '-.']
    
    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fontsize_dict = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (10, 7)  # 增大图形大小，以容纳更多的图例
    fig = {}
    ax_dict_1 = {}
    
    # 动态添加ffnmoe模型配置到字典中
    moe_pattern = re.compile(r'E(\d+)T(\d+)')
    cnnffnmoe_pattern = re.compile(r'E(\d+)T(\d+)')
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        mask = metric_name not in ['Loss'] and stat == 'mean'
        if mask:
            # 获取模型名称
            model_name = df_name_list[1]
            
            # 处理带有ExxTxx格式的模型名称
            moe_match = None
            if len(df_name_list) > 2 and df_name_list[1] == 'ffnmoe':
                # 查找带有ExxTxx格式的部分
                part = df_name_list[2]
                moe_match = moe_pattern.match(part)
                if moe_match:
                    experts = part.split('T')[0][1:]
                    top_k = part.split('T')[1]
                    model_name = f'ffnmoe_E{experts}T{top_k}'
            elif len(df_name_list) > 2 and df_name_list[1] == 'cnnffnmoe':
                part = df_name_list[2]
                moe_match = cnnffnmoe_pattern.match(part)
                if moe_match:
                    experts = part.split('T')[0][1:]
                    top_k = part.split('T')[1]
                    model_name = f'cnnffnmoe_E{experts}T{top_k}'
                    
            # 如果是ffnmoe模型，为其分配标签、颜色、线型
            if 'ffnmoe' in model_name or 'cnnffnmoe' in model_name:
                # 提取E和T的值
                if moe_match:
                    experts, top_k = moe_match.groups()
                else:
                    parts = model_name.split('_')
                    for part in parts:
                        if part.startswith('E') and 'T' in part:
                            experts = part.split('T')[0][1:]
                            top_k = part.split('T')[1]
                            break
                    else:
                        # 如果没有找到ExxTxx格式，尝试其他格式
                        if len(parts) >= 3 and parts[-2].startswith('E') and parts[-1].startswith('T'):
                            experts = parts[-2][1:]
                            top_k = parts[-1][1:]
                        else:
                            # 默认值
                            experts = "?"
                            top_k = "?"
                
                # 为ffnmoe模型添加标签
                if model_name not in label_dict:
                    label_dict[model_name] = f'{model_name} (E={experts},T={top_k})'
                
                # 为ffnmoe模型分配颜色、线型
                if model_name not in color_dict:
                    # 计算索引以循环使用颜色
                    idx = len(color_dict) % len(moe_colors)
                    color_dict[model_name] = moe_colors[idx]
                
                if model_name not in linestyle_dict:
                    # 计算索引以循环使用线型
                    idx = len(linestyle_dict) % len(moe_linestyles)
                    linestyle_dict[model_name] = moe_linestyles[idx]
            
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            
            # 重要变更：确保fig_name只基于数据集和指标，不包含模型名称
            # 这样所有模型将绘制在同一张图上
            fig_name = '_'.join([df_name_list[0], metric_name, stat])
            
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            y = df_history[df_name].iloc[0].to_numpy()
            y_err = df_history[df_name_std].iloc[0].to_numpy() if df_name_std in df_history else np.zeros_like(y)
            x = np.arange(len(y))
            xlabel = 'Epoch'
            pivot = model_name
            ylabel = metric_name
            
            # 使用分配给模型的标签、颜色、线型
            # 如果字典中没有对应的值，使用默认值
            model_label = label_dict.get(pivot, pivot)
            model_color = color_dict.get(pivot, 'black')
            model_linestyle = linestyle_dict.get(pivot, '-')
            
            model_label_rename={'resnetl18': 'Resnet w linear', 'resnetm18': 'Resnet w moe'}
            model_label = model_label_rename.get(pivot, model_label)
            # y中_err异常值移除（某个点大于err的三倍均值的），替换为前一点的值
            y_err_mean = np.mean(y_err)
            y_err_std = np.std(y_err)
            y_err_threshold = 3 * y_err_std
            for i in range(len(y_err)):
                if y_err[i] > y_err_mean + y_err_threshold:
                    y_err[i] = y_err[i-1] if i > 0 else y_err[i+1] if len(y_err) > 1 else y_err[i]
                    y[i] = y[i-1] if i > 0 else y[i+1] if len(y) > 1 else y[i]
                    
            # # 如果y中存在和附近点的差值大于附近其他点之间的3倍（即极大的不稳定，与yerr无关），则替换为附近点的值
            # window_size = 5  # 使用5个点的滑动窗口
            # for i in range(len(y)):
            #     if i < window_size//2 or i >= len(y) - window_size//2:
            #         continue
            #     window = y[i-window_size//2:i+window_size//2+1]
            #     diffs = np.abs(np.diff(window))
            #     mean_diff = np.mean(diffs)
            #     curr_diff = np.abs(y[i] - np.mean([y[i-1], y[i+1]]))
            #     if curr_diff >  2*mean_diff:
            #         y[i] = np.mean([y[i-1], y[i+1]])
            
            # 移除marker参数，仅使用线型和颜色区分
            ax_1.plot(x, y, label=model_label, color=model_color,
                      linestyle=model_linestyle)
            ax_1.fill_between(x, (y - y_err), (y + y_err), color=model_color, alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict.get(metric_name, 'best'), fontsize=fontsize_dict['legend'])
            ax_1.set_title(f'{df_name_list[0]} {metric_name} ', fontsize=fontsize_dict['label'])
    
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        dir_name = 'lc'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.tight_layout()
        # 调整图例位置，可能需要调整以适应更多的模型
        ax_dict_1[fig_name].legend(loc='best', fontsize=fontsize_dict['legend'], bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
