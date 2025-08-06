import joblib
import tensorflow as tf
from typing import Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

from root_utils.config import MODEL_SAVE_DIR

class ModelService:
    def __init__(self, models_dir: Path = MODEL_SAVE_DIR):
        self.models = {}
        self.models_dir = models_dir
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True)
        self.loaded_models = {}  # 新增：缓存已加载模型
        self.feature_scalers = {}  # 存储每个机型的特征缩放器
        self.target_scalers = {}  # 存储每个机型的目标缩放器

    def load_model(self, aircraft_type: str):
        """加载指定机型的模型"""
        # 检查模型文件路径
        model_dir = self.models_dir / aircraft_type
        model_files = list(model_dir.glob("*.h5"))

        if not model_files:
            raise FileNotFoundError(f"找不到{aircraft_type}的模型文件")

        model_file = model_files[0]

        # 检查是否已加载相同的模型
        if aircraft_type in self.models and self.models[aircraft_type].filepath == str(model_file):
            print(f"ℹ️ 从缓存加载{aircraft_type}模型")
            self.models[aircraft_type] = self.loaded_models[aircraft_type]
            return

        try:
            # 实际加载模型
            self.models[aircraft_type] = tf.keras.models.load_model(model_file)
            print(f"✅ 成功加载 {aircraft_type} 模型: {model_file.name}")

            # 加载对应的缩放器
            self.load_scalers(aircraft_type)

        except Exception as e:
            if aircraft_type in self.models:
                del self.models[aircraft_type]
            raise RuntimeError(f"加载 {aircraft_type} 模型失败: {str(e)}")
        # 加载成功后添加到缓存
        self.loaded_models[aircraft_type] = self.models[aircraft_type]

    def load_scalers(self, aircraft_type: str):
        """加载指定机型的特征和目标缩放器"""
        scaler_dir = self.models_dir / aircraft_type
        if not scaler_dir.exists():
            raise FileNotFoundError(f"机型目录不存在: {scaler_dir}")

        # 特征缩放器加载
        x_scaler_path = scaler_dir / "x_scaler.pkl"
        if x_scaler_path.exists():
            try:
                with open(x_scaler_path, 'rb') as f:
                    self.feature_scalers[aircraft_type] = joblib.load(f)
                print(f"成功加载 {aircraft_type} 特征缩放器")
            except Exception as e:
                print(f"加载特征缩放器失败: {str(e)}")
                self.feature_scalers[aircraft_type] = None
        else:
            print(f"警告: 特征缩放器文件不存在 {x_scaler_path}")
            self.feature_scalers[aircraft_type] = None

        # 目标缩放器加载
        y_scaler_path = scaler_dir / "y_scaler.pkl"
        if y_scaler_path.exists():
            try:
                with open(y_scaler_path, 'rb') as f:
                    self.target_scalers[aircraft_type] = joblib.load(f)
                print(f"成功加载 {aircraft_type} 目标缩放器")
            except Exception as e:
                print(f"加载目标缩放器失败: {str(e)}")
                self.target_scalers[aircraft_type] = None
        else:
            print(f"警告: 目标缩放器文件不存在 {y_scaler_path}")
            self.target_scalers[aircraft_type] = None

    def predict(self, aircraft_type: str, input_data: Dict[str, Any]) -> float:
        """
        执行噪声预测
        :param aircraft_type: 机型(A320/B738)
        :param input_data: 包含所有输入参数的字典
        必需输入字段:
        dateTime: 时间戳 (e.g. "2023-01-01 12:00:00")
        Course: 航向角度 (0-360)
        Speed: 速度 (km/h)
        Altitude: 高度 (m)
        WindSpeed: 风速 (m/s)
        AirTemperature: 气温 (°C)
        :return: 预测噪声值(dB)
        """
        # 确保机型模型存在
        if aircraft_type not in self.models:
            self.load_model(aircraft_type)

        # 确保缩放器已加载
        if aircraft_type not in self.feature_scalers:
            self.load_scalers(aircraft_type)

        # 使用机型特定的缩放器
        feature_scaler = self.feature_scalers.get(aircraft_type)
        target_scaler = self.target_scalers.get(aircraft_type)

        if not feature_scaler or not target_scaler:
            raise ValueError(f"{aircraft_type} 机型的缩放器未正确加载")

        # 准备特征并缩放
        features = self._prepare_features(input_data)
        scaled_features = feature_scaler.transform(features.reshape(1, -1))

        # 重塑为模型需要的三维输入
        # 调整形状以匹配模型期望的 (None, 40, 17)
        scaled_features = np.repeat(scaled_features, 40, axis=1).reshape(1, 40, -1)

        # 执行预测并逆缩放
        prediction_scaled = self.models[aircraft_type].predict(scaled_features)
        prediction = target_scaler.inverse_transform(prediction_scaled)
        return prediction[0][0]

    def _prepare_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        优化版特征工程，避免创建完整DataFrame，直接计算特征值
        参数: input_data: 包含所有输入参数的字典
        返回: np.ndarray: 特征向量(1D数组)
        """
        # 确保有日期时间字段，使用当前时间作为默认
        if 'dateTime' not in input_data:
            input_data['dateTime'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将日期字符串转换为datetime对象
        dt = pd.to_datetime(input_data['dateTime'])

        # 计算时间特征
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        second_of_day = hour * 3600 + minute * 60 + second
        day_of_year = dt.dayofyear

        # 向量化计算替代重复操作
        feature_vector = np.array([
            # 方向特征
            np.sin(np.radians(input_data.get('Course', 0))),
            np.cos(np.radians(input_data.get('Course', 0))),

            # 速度/高度比率
            input_data.get('Speed', 0) / (input_data.get('Altitude', 1) + 1e-5),

            # 时间周期特征
            np.sin(second_of_day * (2 * np.pi / 86400)),  # 日周期正弦
            np.cos(second_of_day * (2 * np.pi / 86400)),  # 日周期余弦
            np.sin((day_of_year - 1) * (2 * np.pi / 365.25)),  # 年周期正弦
            np.cos((day_of_year - 1) * (2 * np.pi / 365.25)),  # 年周期余弦
            np.sin(minute * (2 * np.pi / 60)),  # 分钟周期正弦
            np.cos(minute * (2 * np.pi / 60)),  # 分钟周期余弦
            np.sin(second * (2 * np.pi / 60)),  # 秒周期正弦
            np.cos(second * (2 * np.pi / 60)),  # 秒周期余弦

            # 环境特征交互
            input_data.get('WindSpeed', 0) * input_data.get('AirTemperature', 0),
            input_data.get('Distance', 0),
            input_data.get('RelativeHumidity', 0),
            day_of_year,
            hour,
            minute
        ])
        return feature_vector

    # 在model_service中扩展
    def predict_and_visualize(self, aircraft_type, input_data):
        prediction = self.predict(aircraft_type, input_data)

        # 生成可视化图像
        image_path = generate_noise_visualization(
            aircraft_type,
            input_data,
            prediction
        )

        return prediction, image_path