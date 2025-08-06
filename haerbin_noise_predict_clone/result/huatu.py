import pandas as pd

import visualization

csv_path = r"D:\Code\python\enhanced_tcn_block\result\A320\Initial\CNN_LSTM_Attention_A320_20_32_16_32_2_64.csv"
output_path = r"D:\Code\Python\enhanced_tcn_block\result\A320\Initial"

metric_csv_path = r"D:\Code\python\enhanced_tcn_block\CNN_LSTM_Attention_A320_res_first.csv"
metric_output_path = r"D:\Code\Python\enhanced_tcn_block\figures\A320\heatmap\Initial"

# 调用绘图函数
visualization.plot_csv(csv_path, output_path)
visualization.plot_error_percentage(csv_path, output_path)
# print("CSV文件路径:", metric_csv_path)
# print("CSV文件内容:")
# print(pd.read_csv(metric_csv_path).head())
#visualization.plot_matrix_heatmap(metric_csv_path, metric_output_path)