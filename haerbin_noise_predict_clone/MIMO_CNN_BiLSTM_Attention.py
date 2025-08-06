import logging
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Bidirectional, Conv1D, Dense, Dropout, Flatten,
    Input, LSTM, MaxPooling1D, Multiply, Permute, RepeatVector
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from root_utils.config import FEATURES, MODEL_SAVE_DIR, PROCESSED_DATA_PATH, SEED
from root_utils.model_manager import eval, get_data, train_and_evaluate_best_model

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


def configure_gpu_memory() -> None:
    """配置GPU显存使用策略"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # 设置显存增长模式
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # 设置显存限制 (根据实际GPU显存调整)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
            )
            logger.info("GPU memory configured successfully")
    except RuntimeError as e:
        logger.error(f"Error configuring GPU memory: {e}")


def set_seeds(seed: int = SEED) -> None:
    """设置随机种子以确保结果可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seeds set to: {seed}")


def set_global_determinism(seed: int = SEED) -> None:
    """设置全局确定性配置"""
    set_seeds(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    logger.info("Global determinism configured")


def cnn_block(inputs, n_filters: int, kernel_size: int, pool_size: int, drop_rate: float) -> tf.Tensor:
    """构建CNN块"""
    x = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(drop_rate)(x)
    return x


def attention_block(inputs) -> tf.Tensor:
    """实现自注意力机制"""
    attention = Dense(1, activation='tanh')(inputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(inputs.shape[-1])(attention)
    attention = Permute([2, 1])(attention)
    weighted_input = Multiply()([inputs, attention])
    return weighted_input


def build_cnn_lstm_attention(
        input_shape: Tuple[int, int],
        output_dim: int,
        n_filters: int,
        kernel_size: int,
        pool_size: int,
        lstm_units: int,
        attention_units: int,
        learning_rate: float = 0.001,
        drop_rate: float = 0.2
) -> Model:
    """构建CNN-LSTM-Attention模型"""
    inputs = Input(shape=input_shape)

    # CNN部分
    x = cnn_block(inputs, n_filters, kernel_size, pool_size, drop_rate)
    x = cnn_block(x, n_filters * 2, kernel_size, pool_size, drop_rate)

    # LSTM部分
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(drop_rate)(x)

    # 注意力机制
    x = attention_block(x)
    x = Flatten()(x)

    # 全连接层
    x = Dense(attention_units, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    logger.info("CNN-LSTM-Attention model built successfully")
    return model


def compile_and_fit(
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        model_name: str,
        batch_size: int,
        epochs: int,
        validation_split: float,
        patience: int = 50,
        loss_plot_path: Optional[Path] = None
) -> Optional[np.ndarray]:
    """编译并训练模型，返回预测结果"""
    set_global_determinism()

    # 确保模型保存目录存在
    model_path = MODEL_SAVE_DIR / f'{model_name}.h5'
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 配置回调函数
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor='val_mean_absolute_error',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_mean_absolute_error',
        patience=patience,
        restore_best_weights=True
    )
    callbacks = [early_stop, checkpoint]

    # 编译模型
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    try:
        # 训练模型
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # 绘制损失曲线
        if loss_plot_path:
            loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(8, 5))
            plt.plot(history.history['loss'], 'red', label='Training loss')
            plt.plot(history.history['val_loss'], 'blue', label='Validation loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(str(loss_plot_path), dpi=400)
            plt.close()
            logger.info(f"Loss plot saved to: {loss_plot_path}")

        # 加载最佳模型权重并返回预测
        model.load_weights(str(model_path))
        y_pred = model.predict(x_test)
        model.summary()
        return y_pred

    except tf.errors.ResourceExhaustedError:
        logger.error("GPU memory exhausted during training. Try reducing batch size or model complexity.")
    except Exception as e:
        logger.exception(f"Training failed: {e}")

    return None


def ensure_directory(path: Path) -> None:
    """确保目录存在，如果不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def train_one_set(
        batch_size: int,
        epochs: int,
        model_name: str,
        time_step_list: List[int],
        filters_list: List[int],
        kernel_size_list: List[int],
        unit_num_list: List[int],
        input_feature_list: List[List[int]],
        test_len: int,
        pred_len: int,
        save: bool,
        aircraft_model: str,
        pool_size_list: List[int] = [2],
        attention_units_list: List[int] = [64]
) -> Dict:
    """训练模型并执行超参数搜索"""
    set_global_determinism()
    train_ratio = 0.9
    best_mse = float('inf')
    best_params = {}

    # 创建目录结构
    figures_dir = Path('./figures')
    result_dir = Path('./result')
    ensure_directory(figures_dir)
    ensure_directory(result_dir)

    # 根据数据文件名确定结果子目录
    processed_data_path = Path(PROCESSED_DATA_PATH)
    if 'first.csv' in processed_data_path.name:
        sub_dir_name = "Initial"
    elif 'second.csv' in processed_data_path.name:
        sub_dir_name = "Improved"
    else:
        logger.error(f"Invalid data file name: {processed_data_path.name}")
        return best_params

    # 机型专属目录
    aircraft_prediction_dir = figures_dir / aircraft_model / "prediction" / sub_dir_name
    aircraft_heatmap_dir = figures_dir / aircraft_model / "heatmap" / sub_dir_name
    aircraft_result_dir = result_dir / aircraft_model / sub_dir_name

    ensure_directory(aircraft_prediction_dir)
    ensure_directory(aircraft_heatmap_dir)
    ensure_directory(aircraft_result_dir)

    # 超参数搜索
    for time_step in time_step_list:
        for filters in filters_list:
            for kernel_size in kernel_size_list:
                for unit_num in unit_num_list:
                    for pool_size in pool_size_list:
                        for attention_units in attention_units_list:
                            for input_index in input_feature_list:
                                input_size = len(input_index)

                                # 验证特征索引
                                if any(i >= len(FEATURES) for i in input_index):
                                    logger.error(f"Invalid feature indices: {input_index}")
                                    continue

                                input_features = [FEATURES[i] for i in input_index]

                                # 创建文件名
                                params_str = f"{model_name}_{aircraft_model}_{time_step}_{filters}_{kernel_size}_{unit_num}_{pool_size}_{attention_units}"
                                fig_path = aircraft_prediction_dir / f"{params_str}.png"
                                loss_path = aircraft_prediction_dir / f"{params_str}_loss.png"
                                csv_path = aircraft_result_dir / f"{params_str}.csv"

                                # 获取数据
                                data = get_data(
                                    time_step, test_len, pred_len, train_ratio,
                                    input_features, aircraft_model
                                )
                                if any(d is None for d in data[:4]):
                                    logger.warning(f"Skipping parameters due to data issues: {params_str}")
                                    continue

                                train_seq_x, train_seq_y, test_seq_x, test_seq_y, y_scaler, test_indices = data

                                # 构建模型
                                model = build_cnn_lstm_attention(
                                    (time_step, input_size),
                                    pred_len,
                                    n_filters=filters,
                                    kernel_size=kernel_size,
                                    pool_size=pool_size,
                                    lstm_units=unit_num,
                                    attention_units=attention_units
                                )

                                # 训练模型
                                predictions = compile_and_fit(
                                    model,
                                    np.array(train_seq_x),
                                    np.array(train_seq_y),
                                    np.array(test_seq_x),
                                    f"{model_name}_{aircraft_model}",
                                    batch_size,
                                    epochs,
                                    1 - train_ratio,
                                    loss_plot_path=loss_path
                                )

                                if predictions is None:
                                    logger.warning(f"Training failed for parameters: {params_str}")
                                    continue

                                # 后处理预测结果
                                predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
                                ground_truth = y_scaler.inverse_transform(np.array(test_seq_y).reshape(-1, 1)).reshape(
                                    -1)
                                predictions = np.round(predictions, 2)
                                ground_truth = np.round(ground_truth, 2)

                                # 保存结果到CSV
                                ensure_directory(csv_path.parent)
                                results_df = pd.DataFrame({"real": ground_truth, "pred": predictions})
                                results_df.to_csv(csv_path, index=False)

                                # 评估性能
                                mae, mse, mape, r2 = eval(
                                    predictions, ground_truth, pred_len, test_len,
                                    draw=True, fig_name=str(fig_path), save_csv=False,
                                    test_indices=test_indices,
                                    original_data_path=str(PROCESSED_DATA_PATH),
                                    aircraft_model=aircraft_model
                                )

                                # 记录结果
                                result_data = {
                                    'mape': [mape],
                                    'mae': [mae],
                                    'mse': [mse],
                                    'r2': [r2],
                                    'time_step': [time_step],
                                    'filters': [filters],
                                    'kernel_size': [kernel_size],
                                    'unit_num': [unit_num],
                                    'pool_size': [pool_size],
                                    'attention_units': [attention_units],
                                    'input_feature': ['_'.join(input_features)]
                                }

                                # 保存到结果文件
                                result_file = f"./{model_name}_{aircraft_model}_res_{sub_dir_name.lower()}.csv"
                                df = pd.DataFrame(result_data)
                                df.to_csv(result_file, mode='a', header=not Path(result_file).exists())

                                # 更新最佳参数
                                if mse < best_mse:
                                    best_mse = mse
                                    best_params = {
                                        'time_step': time_step,
                                        'filters': filters,
                                        'kernel_size': kernel_size,
                                        'unit_num': unit_num,
                                        'pool_size': pool_size,
                                        'attention_units': attention_units,
                                        'input_feature': input_features
                                    }
                                    logger.info(f"New best MSE: {mse:.4f} with params: {best_params}")

    return best_params


if __name__ == '__main__':
    configure_gpu_memory()

    # 配置超参数
    dry_run = False
    if dry_run:
        epochs = 10
        time_step_list = [30]
        unit_num_list = [16]
        filters_list = [16]
        kernel_size_list = [4]
        pool_size_list = [2]
        attention_units_list = [64]
        input_feature_list = [list(range(len(FEATURES)))]
    else:
        epochs = 50
        time_step_list = [20, 30, 40]
        filters_list = [32, 64]
        kernel_size_list = [4, 8, 12]
        unit_num_list = [16, 32, 64]
        pool_size_list = [2, 4]
        attention_units_list = [64]
        input_feature_list = [list(range(len(FEATURES)))]

    batch_size = 8
    test_len = 50
    pred_len = 1
    save = True

    # 获取机型输入
    aircraft_model = input("Please enter the Aircraft_model: ").strip()
    if not aircraft_model:
        logger.error("Aircraft model cannot be empty")
        exit(1)

    model_name = 'CNN_LSTM_Attention'

    # 训练模型
    best_params = train_one_set(
        batch_size, epochs, model_name,
        time_step_list, filters_list, kernel_size_list, unit_num_list, input_feature_list,
        test_len, pred_len, save, aircraft_model
    )

    # 输出最佳参数
    logger.info("Best model parameters:")
    for param, value in best_params.items():
        logger.info(f"{param}: {value}")

    # 训练最终模型
    best_model = train_and_evaluate_best_model(
        model_name, aircraft_model, best_params,
        batch_size, epochs, test_len, pred_len
    )

    # 坏点处理
    from result.BadPointElimination import eliminate_bad_points

    new_data_path = eliminate_bad_points(aircraft_model)
    logger.info(f"Bad points processed. New data at: {new_data_path}")