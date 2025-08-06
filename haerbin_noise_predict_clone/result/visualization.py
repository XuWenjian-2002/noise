import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, MultipleLocator

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_csv(csv_path, output_path=None, x_column=None, y_columns=None,
             title=None, x_label=None, y_label=None,
             figsize=(10, 6), colors=None, style='-',
             grid=True, legend=True, dpi=300, legend_fontsize=12):
    df = pd.read_csv(csv_path)
    if x_column is None:
        x_column = df.columns[0]
    if y_columns is None:
        y_columns = [col for col in df.columns if col != x_column]
    if x_label is None:
        x_label = x_column

    plt.figure(figsize=figsize)
    x_data = df[x_column] + 1

    # 添加刻度控制
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))  # 控制最多显示10个刻度

    for i, col in enumerate(y_columns):
        color = colors[i] if colors and i < len(colors) else None
        label = col.capitalize() if col in ['real', 'pred'] else col
        plt.plot(x_data, df[col], label=label, color=color, linestyle=style)

    if 'real' in y_columns:
        real_mean = df['real'].mean()
        plt.axhline(y=real_mean, linestyle='--', color='blue', label=f'Real Mean: {real_mean:.2f}')
    if 'pred' in y_columns:
        pred_mean = df['pred'].mean()
        plt.axhline(y=pred_mean, linestyle='--', color='red', label=f'Pred Mean: {pred_mean:.2f}')

    plt.title(title, fontsize=20)
    plt.xlabel('Round', fontsize=16, fontweight='bold')
    plt.ylabel('Value', fontsize=16, fontweight='bold')
    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)
    if legend:
        plt.legend(prop={'size': legend_fontsize})

    plt.tight_layout()

    if output_path is not None:
        if os.path.isdir(output_path):
            base_name = os.path.basename(csv_path)
            file_name, _ = os.path.splitext(base_name)
            output_path = os.path.join(output_path, f"{file_name}_predicated.png")
    else:
        output_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_predicated.png")

    plt.savefig(output_path, dpi=dpi)
    print(f"图表已保存至: {output_path}")
    plt.close()


def plot_error_percentage(csv_path, output_path=None,
                          real_col='real', pred_col='pred',
                          title=None, x_label='Round', y_label='Error Percentage (%)',
                          figsize=(10, 6), color='skyblue',
                          grid=True, dpi=300, fontsize=12):
    df = pd.read_csv(csv_path)
    if real_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"CSV文件中必须包含 '{real_col}' 和 '{pred_col}' 列")

    df['error_percentage'] = ((df[pred_col] - df[real_col]) / df[real_col]) * 100
    plt.figure(figsize=figsize)
    plt.bar(df.index, df['error_percentage'], color=color)

    plt.xlabel(x_label, fontsize=fontsize + 2, fontweight='bold')
    plt.ylabel(y_label, fontsize=fontsize + 2, fontweight='bold')

    ax = plt.gca()
    # 设置每十轮一个刻度
    ax.xaxis.set_major_locator(MultipleLocator(10))

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if grid:
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_path is not None:
        if os.path.isdir(output_path):
            base_name = os.path.basename(csv_path)
            file_name, _ = os.path.splitext(base_name)
            output_path = os.path.join(output_path, f"{file_name}_error_percentage.png")
    else:
        output_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_error_percentage.png")

    plt.savefig(output_path, dpi=dpi)
    print(f"误差百分比图表已保存至: {output_path}")
    plt.close()


def plot_matrix_heatmap(csv_path, output_path=None,
                        title='MSE',
                        x_label='Kernel Size', y_label='Filters',
                        top_label='Time Step', right_label='Unit Num',
                        figsize=(8, 6),
                        cmap='white_to_purple',  # 修改默认颜色映射名称
                        dpi=300, fontsize=14,
                        annotate=True, fmt='.2f',
                        index_col=None,
                        vmin=None, vmax=None):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 调试信息：打印实际列名
    print("CSV文件中的列名:", df.columns.tolist())

    # 尝试匹配列名（不区分大小写和空格）
    col_mapping = {
        'mse': None, 'time_step': None,
        'filters': None, 'kernel_size': None,
        'unit_num': None
    }

    # 查找匹配的列名
    for col in df.columns:
        col_lower = col.strip().lower()
        for key in col_mapping:
            if key in col_lower or col_lower in key:
                col_mapping[key] = col
                break

    # 检查是否找到所有必需列
    missing_cols = [key for key, val in col_mapping.items() if val is None]
    if missing_cols:
        raise ValueError(f"CSV文件中缺少以下列: {', '.join(missing_cols)}")

    print("映射的列名:", col_mapping)

    # 重命名列以便后续处理
    df = df.rename(columns={
        col_mapping['mse']: 'mse',
        col_mapping['time_step']: 'time_step',
        col_mapping['filters']: 'filters',
        col_mapping['kernel_size']: 'kernel_size',
        col_mapping['unit_num']: 'unit_num'
    })

    # 转换数据类型
    df = df[['mse', 'time_step', 'filters', 'kernel_size', 'unit_num']].apply(
        pd.to_numeric, errors='coerce'
    )

    # 检查数据有效性
    if df.empty:
        raise ValueError("CSV文件无有效数据行")
    if df.isna().any().any():
        raise ValueError("数据包含缺失值，请检查")

    # 确定颜色条范围
    vmin = df['mse'].min() if vmin is None else vmin
    vmax = df['mse'].max() if vmax is None else vmax

    # 创建自定义颜色映射：从白色到亮紫色
    if cmap == 'white_to_purple':
        # 创建从白色(#FFFFFF)到亮紫色(#AA00AA)的渐变
        colors = ["#FFFFFF", "#E0B0FF", "#C060FF", "#AA00AA"]
        cmap = mcolors.LinearSegmentedColormap.from_list("white_to_purple", colors)
    elif isinstance(cmap, str):
        # 如果是字符串，使用matplotlib内置的颜色映射
        cmap = plt.get_cmap(cmap)

    # 处理输出路径
    if output_path is not None and os.path.isdir(output_path):
        base_name = os.path.basename(csv_path)
        file_name = os.path.splitext(base_name)[0]
    else:
        output_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path)
        file_name = os.path.splitext(base_name)[0]
        output_path = output_dir

    # 为每一行数据创建热力图
    for i, row in df.iterrows():
        fig, ax = plt.subplots(figsize=figsize)
        data = [[row['mse']]]
        im = ax.imshow(np.array(data, dtype=float),
                       cmap=cmap,
                       aspect='auto',
                       vmin=vmin,
                       vmax=vmax)


        # 设置x轴标签（底部）- 水平显示
        ax.set_xticks([0])
        ax.set_xticklabels([f"{int(row['kernel_size'])}"],
                            fontsize=fontsize,
                            fontweight='bold',
                            rotation=0)  # 确保水平显示

        # 设置y轴标签（左侧）- 垂直显示
        ax.set_yticks([0])
        ax.set_yticklabels([f"{int(row['filters'])}"],
                            fontsize=fontsize,
                            fontweight='bold',
                            rotation=90,  # 垂直显示
                            va='center')  # 垂直居中

        # 添加标签文本（不包含变量名）
        # 顶部标签（水平显示）
        ax.annotate(f"{int(row['time_step'])}",
                    xy=(0.5, 1.05), xycoords='axes fraction',
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight='bold')

        # 右侧标签（垂直显示，放置在图片和颜色条中间）
        ax.annotate(f"{int(row['unit_num'])}",
                    xy=(1.12, 0.5), xycoords='axes fraction',
                    ha='center', va='center',
                    rotation=90,  # 垂直显示
                    fontsize=fontsize, fontweight='bold')

        if annotate:
            ax.text(0, 0, f"{row['mse']:{fmt}}",
                    ha='center', va='center',
                    color='black', fontsize=fontsize*2, fontweight='bold')

        # 添加轴标题（移除了变量名，只显示数值）
        ax.set_xlabel(x_label, fontsize=fontsize, fontweight='bold', labelpad=10)
        ax.set_ylabel(y_label, fontsize=fontsize, fontweight='bold', labelpad=25, rotation=90)

        # 添加顶部和右侧标题（移除了变量名）
        ax.annotate(top_label,
                    xy=(0.5, 1.15), xycoords='axes fraction',
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight='bold')

        ax.annotate(right_label,
                    xy=(1.15, 0.5), xycoords='axes fraction',
                    ha='center', va='center', rotation=90,
                    fontsize=fontsize, fontweight='bold')

        # 创建颜色条（移除了"MSE"标题）
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
        cbar.ax.tick_params(labelsize=fontsize)

        plt.tight_layout()
        row_output_path = os.path.join(output_path, f"{file_name}_round_{i + 1}_mse_heatmap.png")
        plt.savefig(row_output_path, dpi=dpi, bbox_inches='tight')
        print(f"第 {i + 1} 轮热力图已保存至: {row_output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='从CSV文件绘制图表')
    parser.add_argument('csv_path', help='CSV文件路径')
    parser.add_argument('--plot-type', choices=['line', 'error_bar', 'metrics_heatmap', 'matrix_heatmap'],
                        default='line',
                        help='图表类型: line=折线图, error_bar=误差百分比柱状图, metrics_heatmap=指标热力图, matrix_heatmap=矩阵热力图')
    parser.add_argument('--output', '-o', help='输出图片路径')
    parser.add_argument('--x_column', '-x', help='X轴列名（折线图用）')
    parser.add_argument('--y_columns', '-y', nargs='+', help='Y轴列名列表（折线图用）')
    parser.add_argument('--title', '-t', help='图表标题')
    parser.add_argument('--x_label', help='X轴标签')
    parser.add_argument('--y_label', help='Y轴标签')
    parser.add_argument('--figsize', nargs='2', type=int, default=[10, 6], help='图表尺寸')
    parser.add_argument('--colors', nargs='+', help='折线颜色列表（折线图用）')
    parser.add_argument('--style', default='-', help='线条样式（折线图用）')
    parser.add_argument('--no_grid', action='store_false', dest='grid', help='不显示网格')
    parser.add_argument('--no_legend', action='store_false', dest='legend', help='不显示图例（折线图用）')
    parser.add_argument('--dpi', type=int, default=300, help='输出图片分辨率')
    parser.add_argument('--legend_fontsize', type=int, default=12, help='图例文字字体大小（折线图用）')
    parser.add_argument('--real_col', default='real', help='真实值列名（误差图用）')
    parser.add_argument('--pred_col', default='pred', help='预测值列名（误差图用）')
    parser.add_argument('--bar_color', default='skyblue', help='柱状图颜色（误差图用）')
    parser.add_argument('--fontsize', type=int, default=12, help='字体大小（误差图用）')
    parser.add_argument('--cmap', default='LightPurpule', help='热力图颜色映射')
    parser.add_argument('--annotate', action='store_true', help='在单元格中显示数值')
    parser.add_argument('--no-annotate', dest='annotate', action='store_false', help='不在单元格中显示数值')
    parser.set_defaults(annotate=True)
    parser.add_argument('--fmt', default='.2f', help='数值格式（如.2f、.0f等）')
    parser.add_argument('--index-col', type=int, help='指定作为行索引的列号')
    parser.add_argument('--top-label', default='Time Step', help='顶部坐标轴标签')
    parser.add_argument('--right-label', default='Unit Num', help='右侧坐标轴标签')
    parser.add_argument('--vmin', type=float, help='颜色条最小值')
    parser.add_argument('--vmax', type=float, help='颜色条最大值')

    args = parser.parse_args()

    if args.plot_type == 'line':
        plot_csv(
            args.csv_path,
            output_path=args.output,
            x_column=args.x_column,
            y_columns=args.y_columns,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            figsize=tuple(args.figsize),
            colors=args.colors,
            style=args.style,
            grid=args.grid,
            legend=args.legend,
            dpi=args.dpi,
            legend_fontsize=args.legend_fontsize
        )
    elif args.plot_type == 'error_bar':
        plot_error_percentage(
            args.csv_path,
            output_path=args.output,
            real_col=args.real_col,
            pred_col=args.pred_col,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            figsize=tuple(args.figsize),
            color=args.bar_color,
            grid=args.grid,
            dpi=args.dpi,
            fontsize=args.fontsize
        )
    elif args.plot_type == 'matrix_heatmap':
        plot_matrix_heatmap(
            args.csv_path,
            output_path=args.output,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            top_label=args.top_label,
            right_label=args.right_label,
            figsize=tuple(args.figsize),
            cmap=args.cmap,
            dpi=args.dpi,
            fontsize=args.fontsize,
            annotate=args.annotate,
            fmt=args.fmt,
            index_col=args.index_col,
            vmin=args.vmin,
            vmax=args.vmax
        )


if __name__ == "__main__":
    main()