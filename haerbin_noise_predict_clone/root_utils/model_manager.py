import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import matplotlib

matplotlib.use('Agg')  # 避免GUI依赖
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, save_model
import gc

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 从配置文件导入设置（假设已存在）
try:
    from root_utils.config import PROCESSED_DATA_PATH, TARGET_FEATURE, MODEL_SAVE_DIR, SEED

    logger.info("成功导入配置参数")
except ImportError as e:
    logger.error(f"配置导入失败: {e}")
    raise


def save_model(model, model_name: str) -> None:
    """
    保存模型到指定路径

    Args:
        model: 要保存的模型
        model_name: 模型保存的文件名（带或不带扩展名）

    Raises:
        ValueError: 模型保存失败
    """
    try:
        if not model_name.endswith('.h5'):
            model_name += '.h5'

        save_path = Path(model_name)
        save_model(model, save_path)
        logger.info(f"模型已保存至: {save_path.resolve()}")
    except Exception as e:
        logger.exception(f"保存模型失败: {model_name}")
        raise ValueError(f"模型保存失败: {e}") from e


def load_best_model(model_name: str):
    """
    从指定路径加载模型

    Args:
        model_name: 模型保存的文件名（带或不带扩展名）

    Returns:
        Model: 加载的模型或None（加载失败时）
    """
    try:
        if not model_name.endswith('.h5'):
            model_name += '.h5'

        model_path = Path(model_name)
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return None

        model = load_model(model_path)
        logger.info(f"模型已从 {model_path.resolve()} 加载")
        return model
    except Exception as e:
        logger.exception(f"加载模型失败: {model_name}")
        return None


def apply_feature_weights(data: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    根据特征权重加权特征

    Args:
        data: 原始数据DataFrame
        weights: 特征权重字典 {特征名: 权重值}

    Returns:
        pd.DataFrame: 加权后的DataFrame

    Raises:
        KeyError: 指定的特征在数据中不存在
    """
    weighted_data = data.copy()
    for feature, weight in weights.items():
        if feature not in weighted_data.columns:
            logger.warning(f"特征 '{feature}' 不在数据列中，跳过加权")
            continue
        weighted_data[feature] = weighted_data[feature] * weight
    return weighted_data


def read_raw_data(aircraft_model: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    读取原始数据并进行基本验证

    Args:
        aircraft_model: 可选，指定要筛选的飞机型号

    Returns:
        pd.DataFrame: 读取的数据，或None（失败时）
    """
    try:
        # 验证数据文件存在
        if not PROCESSED_DATA_PATH.exists():
            logger.error(f"处理数据文件不存在: {PROCESSED_DATA_PATH}")
            return None

        raw_data = pd.read_csv(PROCESSED_DATA_PATH, header=0)

        # 机型筛选逻辑
        if aircraft_model:
            if 'Aircraft_model' not in raw_data.columns:
                logger.error("数据中缺少'Aircraft_model'列")
                return None

            filtered_data = raw_data[raw_data['Aircraft_model'] == aircraft_model]

            if filtered_data.empty:
                logger.warning(f"未找到机型 '{aircraft_model}' 的数据")
                return None

            logger.info(f"成功筛选机型 '{aircraft_model}'，数据量: {len(filtered_data)}")
            return filtered_data

        logger.info(f"成功读取数据，总样本量: {len(raw_data)}")
        return raw_data

    except Exception as e:
        logger.exception("读取原始数据失败")
        return None


def get_data(
        time_step: int,
        test_len: int,
        pred_len: int,
        train_ratio: float,
        input_feature: List[str],
        aircraft_model: Optional[str] = None
) -> Tuple[Optional[np.ndarray], ...]:
    """
    获取并预处理训练和测试数据

    Args:
        time_step: 时间步长
        test_len: 测试集长度
        pred_len: 预测长度
        train_ratio: 训练集比例
        input_feature: 输入特征列表
        aircraft_model: 可选，指定飞机型号

    Returns:
        包含预处理数据的元组，失败时返回None
    """
    try:
        # 1. 读取数据
        all_data = read_raw_data(aircraft_model)
        if all_data is None:
            return (None,) * 6

        # 2. 数据抽样（如果数据量过大）
        if len(all_data) > 50000:
            sample_frac = 0.5
            all_data = all_data.sample(frac=sample_frac, random_state=SEED)
            logger.info(f"数据抽样完成，保留比例: {sample_frac}，抽样后样本量: {len(all_data)}")

        # 3. 验证特征存在
        missing_features = [f for f in input_feature + TARGET_FEATURE if f not in all_data.columns]
        if missing_features:
            logger.error(f"以下特征在数据中缺失: {missing_features}")
            return (None,) * 6

        # 4. 分离特征和目标
        feature_data = all_data[input_feature].values
        target_data = all_data[TARGET_FEATURE].values

        # 5. 数据集划分
        train_num = int(len(feature_data) * train_ratio)
        if train_num <= time_step + pred_len:
            logger.error("训练集样本不足，无法创建时间序列")
            return (None,) * 6

        train_x = feature_data[:train_num]
        train_y = target_data[:train_num]
        test_x = feature_data[train_num:-pred_len]
        test_y = target_data[train_num:-pred_len]

        # 6. 特征标准化
        x_scaler = MinMaxScaler()
        train_x_scaled = x_scaler.fit_transform(train_x)
        test_x_scaled = x_scaler.transform(test_x)

        # 7. 目标变量标准化
        y_scaler = MinMaxScaler()
        train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1))
        test_y_scaled = y_scaler.transform(test_y.reshape(-1, 1))

        # 8. 生成序列数据
        def create_sequences(data_x, data_y, start_idx, end_idx):
            sequences_x, sequences_y = [], []
            for k in range(len(data_x) - time_step):
                if k + time_step + pred_len > len(data_y):
                    continue
                x = data_x[k:k + time_step]
                y = data_y[k + time_step:k + time_step + pred_len]
                sequences_x.append(x)
                sequences_y.append(y)
            return sequences_x, sequences_y

        train_seq_x, train_seq_y = create_sequences(train_x_scaled, train_y_scaled, 0, train_num)
        test_seq_x, test_seq_y = create_sequences(test_x_scaled, test_y_scaled, train_num, len(feature_data))

        # 9. 创建测试集索引
        test_indices = [train_num + k + time_step for k in range(len(test_x_scaled) - time_step)]

        # 10. 机型专属处理
        if aircraft_model:
            model_dir = Path(MODEL_SAVE_DIR) / aircraft_model
            model_dir.mkdir(parents=True, exist_ok=True)

            # 保存缩放器
            joblib.dump(x_scaler, model_dir / "x_scaler.pkl")
            joblib.dump(y_scaler, model_dir / "y_scaler.pkl")
            logger.info(f"为机型 '{aircraft_model}' 保存缩放器")

        # 11. 内存优化
        del_vars = [
            'train_x_scaled', 'test_x_scaled', 'train_y_scaled', 'test_y_scaled',
            'feature_data', 'target_data', 'all_data', 'train_x', 'train_y', 'test_x', 'test_y'
        ]
        for var in del_vars:
            if var in locals():
                del locals()[var]

        gc.collect()
        logger.info("中间变量已清除，内存回收完成")

        return (
            np.array(train_seq_x),
            np.array(train_seq_y),
            np.array(test_seq_x),
            np.array(test_seq_y),
            y_scaler,
            test_indices
        )

    except Exception as e:
        logger.exception("数据处理过程中发生错误")
        return (None,) * 6


def _plot_results(
        unique_real_y: List[float],
        unique_pred_y: List[float],
        fig_name: str,
        y_min: float = 65,
        y_max: float = 85
) -> None:
    """
    绘制结果可视化图（封装绘图逻辑）

    Args:
        unique_real_y: 实际值列表
        unique_pred_y: 预测值列表
        fig_name: 图像保存路径
        y_min: Y轴最小值
        y_max: Y轴最大值
    """
    plt.figure(figsize=(12, 8))
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('值', fontsize=16)

    # 绘制折线图
    plt.plot(unique_real_y, c='green', marker='*', ms=1, alpha=0.75, label='实际值')
    plt.plot(unique_pred_y, c='red', marker='o', ms=1, alpha=0.75, label='预测值')

    # 添加数值标签
    for i, val in enumerate(unique_real_y):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=6, fontweight='bold')
    for i, val in enumerate(unique_pred_y):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=6, fontweight='bold')

    # 设置坐标轴
    plt.xticks(range(len(unique_real_y)))
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.grid(axis='y')
    plt.legend()

    # 确保目录存在
    fig_dir = Path(fig_name).parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(fig_name, dpi=400)
    plt.close()
    logger.info(f"结果可视化图已保存至: {fig_name}")


def eval(
        PreY: Union[np.ndarray, List[float]],
        RealY: Union[np.ndarray, List[float]],
        pred_len: int,
        test_len: int,
        draw: bool = False,
        fig_name: str = 'prediction.png',
        save_csv: bool = False,
        csv_name: str = 'results.csv',
        test_indices: Optional[List[int]] = None,
        original_data_path: Optional[str] = None,
        aircraft_model: Optional[str] = None,
        y_min: float = 65,
        y_max: float = 85
) -> Tuple[float, float, float, float]:
    """
    评估模型性能并可视化结果

    Args:
        PreY: 预测值
        RealY: 实际值
        pred_len: 预测长度
        test_len: 测试长度
        draw: 是否绘制结果图
        fig_name: 结果图保存路径
        save_csv: 是否保存CSV结果
        csv_name: CSV保存路径
        test_indices: 测试集索引
        original_data_path: 原始数据路径
        aircraft_model: 飞机型号
        y_min: Y轴最小值
        y_max: Y轴最大值

    Returns:
        评估指标元组 (MAE, MSE, MAPE, R2)
    """
    try:
        # 1. 转换数据格式
        PreY = np.array(PreY).flatten()
        RealY = np.array(RealY).flatten()

        if len(PreY) != len(RealY):
            logger.error(f"预测值({len(PreY)})和实际值({len(RealY)})长度不匹配")
            return (0.0, 0.0, 0.0, 0.0)

        # 2. 计算评估指标
        mae = mean_absolute_error(RealY, PreY)
        mse = mean_squared_error(RealY, PreY)
        mape = mean_absolute_percentage_error(RealY, PreY)
        r2 = r2_score(RealY, PreY)

        logger.info(f"评估结果 - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}")

        # 3. 结果可视化
        if draw:
            # 准备唯一值序列
            unique_real_y = []
            unique_pred_y = []

            all_real_y = RealY.reshape(-1, pred_len)
            all_pred_y = PreY.reshape(-1, pred_len)

            for i in range(min(test_len // pred_len, len(all_real_y))):
                if i < len(all_real_y):
                    unique_real_y.append(all_real_y[i][0])
                if i < len(all_pred_y):
                    unique_pred_y.append(all_pred_y[i][0])

            # 保留两位小数
            unique_real_y = [round(val, 2) for val in unique_real_y]
            unique_pred_y = [round(val, 2) for val in unique_pred_y]

            # 保存CSV
            if save_csv:
                results_df = pd.DataFrame({'true': unique_real_y, 'pred': unique_pred_y})
                Path(csv_name).parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(csv_name, index=False)
                logger.info(f"结果已保存至CSV: {csv_name}")

            # 绘制结果图
            _plot_results(unique_real_y, unique_pred_y, fig_name, y_min, y_max)

            # 4. 收集坏点信息
            bad_points = []
            if test_indices is not None and original_data_path is not None:
                min_length = min(len(test_indices), len(all_real_y))

                for i in range(min_length):
                    error = abs(all_real_y[i][0] - all_pred_y[i][0])
                    if error > 3:  # 误差阈值
                        bad_points.append({
                            'index': test_indices[i],
                            'real': all_real_y[i][0],
                            'pred': all_pred_y[i][0],
                            'error': error
                        })

            # 保存坏点信息
            if bad_points and aircraft_model:
                bad_points_dir = Path(original_data_path).parent / "BadPoints"
                bad_points_dir.mkdir(parents=True, exist_ok=True)
                csv_path = bad_points_dir / f"bad_points_{aircraft_model}.csv"
                pd.DataFrame(bad_points).to_csv(csv_path, index=False)
                logger.info(f"保存{len(bad_points)}个坏点到: {csv_path}")

        return mae, mse, mape, r2

    except Exception as e:
        logger.exception("评估过程中发生错误")
        return (0.0, 0.0, 0.0, 0.0)


def train_and_evaluate_best_model(
        model_name: str,
        aircraft_model: str,
        best_params: Dict,
        batch_size: int,
        epoch: int,
        test_len: int,
        pred_len: int
):
    """
    训练并评估最佳模型

    Args:
        model_name: 模型名称
        aircraft_model: 飞机型号
        best_params: 最佳参数字典
        batch_size: 批次大小
        epoch: 训练轮数
        test_len: 测试长度
        pred_len: 预测长度

    Returns:
        训练好的最佳模型
    """
    try:
        logger.info(f"开始训练最佳模型: {model_name} for {aircraft_model}")

        # 1. 导入模型架构
        from MIMO_CNN_BiLSTM_Attention import CNN_LSTM_Attention, compile_and_fit

        # 2. 获取数据
        input_feature = best_params['input_feature']
        ratio = 0.9
        data = get_data(
            best_params['time_step'], test_len, pred_len, ratio,
            input_feature, aircraft_model
        )

        if any(d is None for d in data[:4]):
            logger.error("数据获取失败，无法训练模型")
            return None

        train_seq_x, train_seq_y, test_seq_x, test_seq_y, y_scaler, test_indices = data

        # 3. 创建模型
        input_shape = (best_params['time_step'], len(input_feature))
        best_model = CNN_LSTM_Attention(
            input_shape,
            pred_len,
            n_filters=best_params['filters'],
            kernel_size=best_params['kernel_size'],
            pool_size=best_params['pool_size'],
            lstm_units=best_params['unit_num'],
            attention_units=best_params['attention_units'],
            lr=0.001,
            drop=0.2
        )

        # 4. 训练模型
        best_model_name = f"{model_name}_{aircraft_model}_best"
        best_model_path = f"./models/{aircraft_model}/{best_model_name}.h5"
        loss_name = f"./figures/{aircraft_model}/prediction/Initial/{best_model_name}_loss.png"

        compile_and_fit(
            best_model,
            np.array(train_seq_x),
            np.array(train_seq_y),
            np.array(test_seq_x),
            best_model_name,
            batch_size,
            epoch,
            1 - ratio,
            50,
            loss_name
        )

        # 5. 保存和加载模型
        save_model(best_model, best_model_path)
        loaded_model = load_best_model(best_model_path)

        if not loaded_model:
            logger.error("模型加载失败")
            return None

        # 6. 预测和评估
        y_pred = loaded_model.predict(np.array(test_seq_x))
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        test_y = y_scaler.inverse_transform(np.array(test_seq_y).reshape(-1, 1)).flatten()

        # 7. 尖峰可视化
        spike_fig_name = f"./figures/{aircraft_model}/prediction/Initial/{model_name}_{aircraft_model}_spike_visual.png"
        eval(
            y_pred, test_y, pred_len, test_len,
            draw=True, fig_name=spike_fig_name,
            save_csv=False, test_indices=test_indices,
            original_data_path=str(PROCESSED_DATA_PATH),
            aircraft_model=aircraft_model
        )

        logger.info(f"最佳模型训练完成: {best_model_name}")
        return best_model

    except ImportError:
        logger.error("无法导入模型架构模块")
        return None
    except Exception as e:
        logger.exception("最佳模型训练过程中出错")
        return None