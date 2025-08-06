import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler

from root_utils.config import DATA_PROCESSED_DIR, FEATURES, RAW_DATA_PATH, TARGET_FEATURE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 字体配置
font_config = {'family': 'SimHei', 'weight': 'bold'}
matplotlib.rc("font", **font_config)

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """数据预处理管道，封装数据清洗和特征工程流程"""

    def __init__(self, raw_data_path: Path, processed_dir: Path):
        """
        初始化数据预处理器

        :param raw_data_path: 原始数据文件路径
        :param processed_dir: 处理后的数据保存目录
        """
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        self.df = None
        self.clean_df = None
        self.original_columns = []

        # 确保输出目录存在
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataPreprocessor for {raw_data_path.name}")

    def load_data(self) -> None:
        """加载原始数据文件"""
        try:
            self.df = pd.read_csv(self.raw_data_path)
            self.original_columns = self.df.columns.tolist()
            logger.info(f"Loaded raw data: {self.raw_data_path.name}")
            logger.info(f"Data shape: {self.df.shape}")
            logger.info(f"Columns: {self.original_columns}")
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def calculate_feature_importance(self) -> None:
        """计算特征重要性并保存可视化结果"""
        try:
            # 确保只包含数值特征
            numerical_features = [
                col for col in FEATURES
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])
            ]

            if not numerical_features:
                logger.warning("No numerical features for importance calculation")
                return

            # 计算相关性
            correlation_matrix = self.df[numerical_features + TARGET_FEATURE].corr()
            feature_importance = correlation_matrix[TARGET_FEATURE[0]].abs().sort_values(ascending=False)

            # 可视化特征重要性
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importance.values, y=feature_importance.index)
            plt.title('特征与Leq的相关性 (CNN-LSTM-Attention)')
            plt.xlabel('相关性绝对值')
            plt.tight_layout()

            # 保存可视化结果
            importance_path = self.processed_dir / "feature_importance.png"
            plt.savefig(importance_path)
            plt.close()
            logger.info(f"Feature importance visualization saved: {importance_path}")
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")

    def detect_outliers(self, eps: float = 0.5, min_samples: int = 10, leq_threshold: float = 75.0) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        使用DBSCAN聚类检测离群值

        :param eps: DBSCAN邻域半径参数
        :param min_samples: 形成核心点所需的最小样本数
        :param leq_threshold: Leq阈值，高于此值不视为离群点
        :return: 清洗后的数据和离群点数据
        """
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return pd.DataFrame(), pd.DataFrame()

        # 验证必要列存在
        required_cols = {'dateTime', 'Leq', 'Distance', 'Aircraft_model'}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            logger.error(f"Missing required columns for outlier detection: {missing_cols}")
            return self.df.copy(), pd.DataFrame()

        try:
            logger.info("Starting outlier detection with DBSCAN...")

            # 转换时间列
            self.df['dateTime'] = pd.to_datetime(self.df['dateTime'], errors='coerce')
            if self.df['dateTime'].isnull().any():
                logger.warning("Some datetime values could not be parsed")

            # 标记高Leq值点
            high_leq_mask = (self.df['Leq'] > leq_threshold)
            logger.info(f"High Leq points identified (> {leq_threshold}): {high_leq_mask.sum()}")

            # 创建时间戳特征
            self.df['timestamp'] = self.df['dateTime'].astype(np.int64) // 10 ** 9

            # 选择聚类特征
            cluster_features = ['Leq', 'Distance', 'Aircraft_model', 'timestamp']
            cluster_df = self.df[cluster_features].copy()

            # 编码机型
            label_encoder = LabelEncoder()
            cluster_df['Aircraft_model'] = label_encoder.fit_transform(cluster_df['Aircraft_model'].astype(str))

            # 标准化特征
            scaler = StandardScaler()
            cluster_scaled = scaler.fit_transform(cluster_df)

            # DBSCAN聚类
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(cluster_scaled)

            # 标记离群点（排除高Leq点）
            is_outlier = (clusters == -1) & (~high_leq_mask)
            outliers = self.df[is_outlier].copy()
            clean_df = self.df[~is_outlier].copy()

            logger.info(f"Original data count: {len(self.df)}")
            logger.info(f"Outliers detected: {len(outliers)}")
            logger.info(f"Cleaned data count: {len(clean_df)}")

            # 可视化聚类结果
            self._visualize_outliers(cluster_scaled, is_outlier, high_leq_mask, eps, min_samples)

            # 保存离群点数据
            outliers_path = self.processed_dir / "outliers_data.csv"
            outliers.to_csv(outliers_path, index=False)
            logger.info(f"Outliers saved: {outliers_path}")

            return clean_df, outliers

        except Exception as e:
            logger.exception("Outlier detection failed")
            return self.df.copy(), pd.DataFrame()

    def _visualize_outliers(self, cluster_scaled: np.ndarray, is_outlier: pd.Series,
                            high_leq_mask: pd.Series, eps: float, min_samples: int) -> None:
        """可视化离群点检测结果"""
        plt.figure(figsize=(12, 8))

        # 正常点
        plt.scatter(
            cluster_scaled[~is_outlier, 0],
            cluster_scaled[~is_outlier, 3],
            c='green', alpha=0.6, label='Normal points'
        )

        # 离群点
        plt.scatter(
            cluster_scaled[is_outlier, 0],
            cluster_scaled[is_outlier, 3],
            c='red', marker='x', label='Outliers'
        )

        # 高Leq点
        plt.scatter(
            cluster_scaled[high_leq_mask, 0],
            cluster_scaled[high_leq_mask, 3],
            c='blue', marker='o', label=f'High Leq (>75)'
        )

        plt.xlabel('Standardized Leq')
        plt.ylabel('Standardized Timestamp')
        plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
        plt.legend()
        plt.tight_layout()

        # 保存可视化结果
        plot_path = self.processed_dir / "outliers_visualization.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Outlier visualization saved: {plot_path}")

    def engineer_features(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """
        执行特征工程

        :param clean_df: 清洗后的数据
        :return: 包含新特征的数据集
        """
        logger.info("Starting feature engineering...")

        try:
            # 创建数据副本避免修改原始数据
            processed_data = clean_df.copy()

            # 统一列名
            column_mapping = {'监测站点': 'Position', '飞机型号': 'Aircraft_model', '日期时间': 'dateTime'}
            processed_data.rename(columns=column_mapping, inplace=True)

            # 处理时间特征
            self._process_time_features(processed_data)

            # 风向特征转换
            self._process_course_features(processed_data)

            # 增加飞机型号特征增强
            self._enhance_aircraft_features(processed_data)

            # 增加物理特征交叉
            self._create_interaction_features(processed_data)

            # 确保所有必要特征存在
            self._ensure_required_features(processed_data)

            logger.info("Feature engineering completed successfully")
            return processed_data

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return clean_df

    def _process_time_features(self, df: pd.DataFrame) -> None:
        """创建基于时间的周期特征"""
        # 确保日期时间列存在且正确解析
        if 'dateTime' not in df.columns or df['dateTime'].isnull().all():
            logger.warning("dateTime column missing or invalid, skipping time features")
            return

        # 基础时间特征
        df['Year'] = df['dateTime'].dt.year
        df['Month'] = df['dateTime'].dt.month
        df['Day'] = df['dateTime'].dt.day
        df['Hour'] = df['dateTime'].dt.hour
        df['Minute'] = df['dateTime'].dt.minute
        df['Second'] = df['dateTime'].dt.second

        # 时间戳特征
        df['timestamp'] = df['dateTime'].astype(np.int64) // 10 ** 9
        df['second_of_day'] = df['Hour'] * 3600 + df['Minute'] * 60 + df['Second']
        df['day_of_year'] = df['dateTime'].dt.dayofyear

        # 周期性编码
        # 一天的周期性
        df['day_cycle_sin'] = np.sin(df['second_of_day'] * (2 * np.pi / 86400))
        df['day_cycle_cos'] = np.cos(df['second_of_day'] * (2 * np.pi / 86400))

        # 一周的周期性
        df['weekday'] = df['dateTime'].dt.weekday
        df['week_cycle_sin'] = np.sin(df['weekday'] * (2 * np.pi / 7))
        df['week_cycle_cos'] = np.cos(df['weekday'] * (2 * np.pi / 7))

        # 一年的周期性
        df['year_cycle_sin'] = np.sin((df['day_of_year'] - 1) * (2 * np.pi / 365.25))
        df['year_cycle_cos'] = np.cos((df['day_of_year'] - 1) * (2 * np.pi / 365.25))

        # 分/秒的周期性
        df['minute_cycle_sin'] = np.sin(df['Minute'] * (2 * np.pi / 60))
        df['minute_cycle_cos'] = np.cos(df['Minute'] * (2 * np.pi / 60))
        df['second_cycle_sin'] = np.sin(df['Second'] * (2 * np.pi / 60))
        df['second_cycle_cos'] = np.cos(df['Second'] * (2 * np.pi / 60))

        logger.info("Time-based features created")

    def _process_course_features(self, df: pd.DataFrame) -> None:
        """处理风向特征"""
        if 'Course' in df.columns:
            df['Course_rad'] = np.radians(df['Course'])
            df['Course_sin'] = np.sin(df['Course_rad'])
            df['Course_cos'] = np.cos(df['Course_rad'])
            logger.info("Course features processed")
        else:
            logger.warning("Course column missing, skipping course features")

    def _enhance_aircraft_features(self, df: pd.DataFrame) -> None:
        """增强飞机型号特征"""
        if 'Aircraft_model' in df.columns:
            model_counts = df['Aircraft_model'].value_counts().to_dict()
            df['Model_Frequency'] = df['Aircraft_model'].map(model_counts)
            df['Model_Rareness'] = 1 / df['Model_Frequency']
            logger.info("Aircraft model features enhanced")
        else:
            logger.warning("Aircraft_model column missing, skipping model enhancement")

    def _create_interaction_features(self, df: pd.DataFrame) -> None:
        """创建特征交互项"""
        # 速度与高度比
        if 'Speed' in df.columns and 'Altitude' in df.columns:
            df['Speed_Altitude_Ratio'] = df['Speed'] / (df['Altitude'] + 1)
        else:
            logger.warning("Speed or Altitude missing, skipping Speed_Altitude_Ratio")

        # 风速与温度交互
        if 'WindSpeed' in df.columns and 'AirTemperature' in df.columns:
            df['Wind_Temperature_Interaction'] = df['WindSpeed'] * df['AirTemperature']
        else:
            logger.warning("WindSpeed or AirTemperature missing, skipping interaction")

    def _ensure_required_features(self, df: pd.DataFrame) -> None:
        """确保所有必要特征存在"""
        # 添加FEATURES中定义的列
        for feature in FEATURES:
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' missing, creating placeholder")
                df[feature] = np.nan

        # 确保目标变量存在
        for target in TARGET_FEATURE:
            if target not in df.columns:
                logger.warning(f"Target '{target}' missing, creating placeholder")
                df[target] = np.nan

        # 确保关键标识列存在
        for col in ['Aircraft_model', 'dateTime', 'Position']:
            if col not in df.columns:
                logger.warning(f"Key column '{col}' missing, creating placeholder")
                df[col] = 'Unknown'

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_all_data_first.csv") -> Path:
        """
        保存处理后的数据

        :param df: 要保存的数据框
        :param filename: 保存的文件名
        :return: 保存的文件路径
        """
        output_path = self.processed_dir / filename

        # 恢复原始列顺序
        valid_original_columns = [col for col in self.original_columns if col in df.columns]
        new_columns = [col for col in df.columns if col not in valid_original_columns]
        final_columns = valid_original_columns + new_columns
        ordered_df = df[final_columns]

        # 保存数据
        ordered_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved: {output_path}")
        logger.info(f"Final data shape: {ordered_df.shape}")

        return output_path


def preprocess_data() -> pd.DataFrame:
    """
    执行完整的数据预处理流程

    :return: 处理后的数据框
    """
    # 确保输出目录存在
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化预处理器
    preprocessor = DataPreprocessor(RAW_DATA_PATH, DATA_PROCESSED_DIR)

    try:
        # 1. 加载数据
        preprocessor.load_data()

        # 2. 计算特征重要性
        preprocessor.calculate_feature_importance()

        # 3. 检测并处理离群点
        clean_df, outliers = preprocessor.detect_outliers()

        # 4. 特征工程
        processed_data = preprocessor.engineer_features(clean_df)

        # 5. 保存处理后的数据
        preprocessor.save_processed_data(processed_data)

        logger.info("\n" + "=" * 50)
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Final processed data size: {len(processed_data)}")

        return processed_data

    except Exception as e:
        logger.exception("Data preprocessing failed")
        return pd.DataFrame()


if __name__ == "__main__":
    # 执行完整的数据预处理流程
    processed_data = preprocess_data()