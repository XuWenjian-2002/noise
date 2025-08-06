# 数据格式化

import pandas as pd


class DataFormatter:
    @staticmethod
    def format_prediction_input(input_data: dict) -> dict:
        """格式化预测输入数据，添加模型需要的额外字段"""
        formatted = input_data.copy()

        # 添加当前时间戳
        formatted['dateTime'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # 添加飞机型号占位符
        formatted['Aircraft_model'] = "Unknown"

        return formatted

    @staticmethod
    def format_prediction_result(prediction: float) -> dict:
        """格式化预测结果"""
        return {
            "noise_level": f"{prediction:.2f} dB",
            "classification": "High" if prediction > 85 else "Moderate" if prediction > 70 else "Low"
        }