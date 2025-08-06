import pandas as pd
import os
from pathlib import Path
from root_utils.config import DATA_PROCESSED_DIR, PROCESSED_DATA_PATH


def eliminate_bad_points(aircraft_model):
    """
    剔除坏点并创建新数据集
    """
    # 原始数据路径
    input_path = str(PROCESSED_DATA_PATH)

    # 坏点文件路径
    bad_points_dir = DATA_PROCESSED_DIR / "BadPoints"
    bad_points_path = bad_points_dir / f"bad_points_{aircraft_model}.csv"

    # 新数据集路径
    output_dir = DATA_PROCESSED_DIR / "CleanedData"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"cleaned_data_{aircraft_model}.csv"

    if not os.path.exists(bad_points_path):
        print(f"未找到坏点文件: {bad_points_path}")
        return None

    try:
        # 读取原始数据和坏点
        original_df = pd.read_csv(input_path)
        bad_points_df = pd.read_csv(bad_points_path)

        # 获取坏点索引
        bad_indices = bad_points_df['index'].unique().tolist()

        # 创建新数据集 (排除坏点)
        cleaned_df = original_df.drop(index=bad_indices, errors='ignore')

        # 保存新数据集
        cleaned_df.to_csv(output_path, index=False)
        print(f"已创建新数据集: {output_path}")
        print(f"原始数据量: {len(original_df)}, 新数据量: {len(cleaned_df)}, 剔除点数: {len(bad_indices)}")

        return output_path

    except Exception as e:
        print(f"处理坏点时出错: {str(e)}")
        return None


if __name__ == "__main__":
    aircraft_model = input("请输入机型名称: ")
    new_data_path = eliminate_bad_points(aircraft_model)

    if new_data_path:
        print(f"新数据集已保存至: {new_data_path}")
        print("请将 app_utils.config.py 中的 PROCESSED_DATA_PATH 更新为:")
        print(f"PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / 'CleanedData' / 'cleaned_data_{aircraft_model}.csv'")