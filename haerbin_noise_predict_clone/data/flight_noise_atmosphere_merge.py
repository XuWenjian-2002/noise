# 基于WGS84椭球体的三维直线距离公式
"""需要先将地理坐标转换为地心地固坐标系（ECEF），再计算欧几里得距离"""

import math
import pandas as pd
from pathlib import Path
import chardet
from functools import reduce


def calculate_distance(lat1, lon1, h1, lat2, lon2, h2):
    # WGS84参数
    a = 6378137.0  # 长半轴 (m)
    b = 6356752.3142  # 短半轴 (m)
    e_sq = 1 - (b ** 2 / a ** 2)  # 偏心率平方

    # 转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 点1的ECEF坐标
    N1 = a / math.sqrt(1 - e_sq * math.sin(lat1_rad) ** 2)
    X1 = (N1 + h1) * math.cos(lat1_rad) * math.cos(lon1_rad)
    Y1 = (N1 + h1) * math.cos(lat1_rad) * math.sin(lon1_rad)
    Z1 = ((1 - e_sq) * N1 + h1) * math.sin(lat1_rad)

    # 点2的ECEF坐标
    N2 = a / math.sqrt(1 - e_sq * math.sin(lat2_rad) ** 2)
    X2 = (N2 + h2) * math.cos(lat2_rad) * math.cos(lon2_rad)
    Y2 = (N2 + h2) * math.cos(lat2_rad) * math.sin(lon2_rad)
    Z2 = ((1 - e_sq) * N2 + h2) * math.sin(lat2_rad)

    # 三维直线距离
    distance = math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2)
    return distance


def convert_and_merge_files(file_paths, output_path, selected_columns, date_format='%Y-%m-%d %H:%M:%S'):
    """
    将文件转换为CSV（若需要）并按列合并

    参数:
        file_paths: 原始文件路径列表（支持CSV/非CSV，非CSV需确保pandas可读取）
        output_path: 合并后的CSV保存路径
        selected_columns: 需要保存的指定字段列表
        date_format: 日期时间列转换格式（可选，仅适用于含dateTime列的文件）
    """
    dfs = []  # 存储各文件DataFrame
    date_sets = []  # 存储各文件有效日期集合
    file_info = []  # 存储文件名和行数信息

    # 第一阶段：读取所有文件并收集数据
    for file_path in file_paths:
        file_path = Path(file_path)
        try:
            # 读取文件逻辑保持不变
            if file_path.suffix == '.csv':
                with open(file_path, 'rb') as f:
                    rawdata = f.read(10000)
                encoding = chardet.detect(rawdata)['encoding']
                df = pd.read_csv(file_path, encoding=encoding)
            else:
                raise ValueError(f"暂不支持该文件格式: {file_path.suffix}")

            # 强制要求必须包含dateTime列
            if 'dateTime' not in df.columns:
                raise ValueError(f"文件 {file_path.name} 缺少dateTime列，无法合并")

            # 转换日期列并收集有效日期
            df['dateTime'] = pd.to_datetime(df['dateTime'], format=date_format, errors='coerce')
            valid_dates = df['dateTime'].dropna()
            date_sets.append(set(valid_dates))

            # 存储文件信息
            file_info.append({
                'name': file_path.name,
                'rows': len(df),
                'valid_dates': len(valid_dates)
            })
            dfs.append(df)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            raise

    # 第二阶段：输出统计信息
    print("\n各文件数据统计:")
    for i, info in enumerate(file_info):
        print(f"文件 {i + 1} ({info['name']}):")
        print(f"  总行数: {info['rows']}")
        print(f"  有效日期数量: {info['valid_dates']}")

    print("\n两两文件日期交集统计:")
    for i in range(len(file_info)):
        for j in range(i + 1, len(file_info)):
            common = len(date_sets[i] & date_sets[j])
            print(f"{file_info[i]['name']} 与 {file_info[j]['name']} 有 {common} 条共同日期")

    # 计算所有文件共有日期
    if date_sets:
        common_dates = reduce(lambda x, y: x & y, date_sets)
        print(f"\n所有文件共有日期数量: {len(common_dates)}")
    else:
        common_dates = set()

    # 第三阶段：合并数据
    try:
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on='dateTime', how='inner'),
            dfs
        ) if dfs else pd.DataFrame()

        # 清理数据（根据需求选择保留或删除）
        merged_df = merged_df.dropna(how='any')

        # 计算监测站点和飞机之间的距离
        merged_df['Distance'] = merged_df.apply(
            lambda row: calculate_distance(row['Lat'], row['Lon'], row['Altitude'], row['SiteLat'], row['SiteLon'],
                                           row['SiteAltitude']), axis=1)

        # 筛选指定的列
        if 'dateTime' not in selected_columns:
            selected_columns = ['dateTime'] + selected_columns
        if 'Distance' not in selected_columns:
            selected_columns = selected_columns + ['Distance']
        merged_df = merged_df[selected_columns]

        # 保存结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)

        print(f"\n合并完成！最终有效数据量: {len(merged_df)}")
        print(f"保存路径: {output_path}")
        return merged_df

    except Exception as e:
        print(f"合并过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    # 保持原有配置不变
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    file1 = r"D:\Code\python\noise\NoiseTraining\data\flight\202503_processed\202503_Tracks_230102342207_flight.csv"
    file2 = r"D:\Code\python\noise\NoiseTraining\data\atmosphere\230102342207\202503_processed\202503_230102342207_atmosphere.csv"
    file3 = r"D:\Code\python\noise\NoiseTraining\data\noise\202503_processed\202503_noise.csv"

    output_file = parent_dir / "data" / "flight_atmosphere_noise_merged_part.csv"

    # 指定需要保存的字段
    selected_columns = [
        'dateTime',
        'Lon', 'Lat', 'Altitude', 'Speed', 'Vspeed', 'Course', # 航迹数据
        '监测站点', '峰值时间', '持续时间', 'Leq', 'SEL', 'Lmax', 'flight_number', 'Aircraft_model', 'SiteLon', 'SiteLat', 'SiteAltitude', # 噪声监测数据
        'AirTemperature', 'BarometricPressure', 'RelativeHumidity', 'WindSpeed' # 气象数据
    ]  # 替换为实际需要的列名

    try:
        convert_and_merge_files([file1, file2, file3], output_file, selected_columns)
    except Exception as e:
        print(f"操作失败：{str(e)}")