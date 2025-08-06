import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目根路径
BASE_DIR = Path(__file__).parent.parent.resolve()
logger.info(f"Project base directory: {BASE_DIR}")

# 模型、数据、页面保存路径
MODEL_SAVE_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "app" / "static"

# 数据路径
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = DATA_RAW_DIR / "flight_atmosphere_noise_merged_part.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "processed_all_data_first.csv"

# 确保目录存在
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_SAVE_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {path}")

# 数据特征配置
TARGET_FEATURE = ['Leq']  # 目标预测变量
FEATURES = [
    'Course_sin', 'Course_cos', 'Speed_Altitude_Ratio',
    'BarometricPressure', 'RelativeHumidity', 'Wind_Temperature_Interaction',
    'day_cycle_sin', 'day_cycle_cos', 'year_cycle_sin', 'year_cycle_cos',
    'minute_cycle_sin', 'minute_cycle_cos', 'second_cycle_sin', 'second_cycle_cos',
    'Distance',
    'Model_Frequency', 'Model_Rareness'
]

SEED = 42
logger.info(f"Configuration loaded with seed: {SEED}")


# 应用配置
APP_CONFIG = {
    "allowed_aircraft_types": ["A320", "B738"],
    "default_site": "徐家窝棚",
    "map_default_zoom": 13
}

# 监测站点配置(与geo_service.py保持一致)
SITE_COORDINATES = {
    'Site1': (31.2304, 121.4737, 10.5),
    'Site2': (31.2356, 121.4789, 12.3)
}


def validate_config() -> None:
    """验证配置完整性"""
    missing_paths = []
    for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH]:
        if not path.exists():
            missing_paths.append(str(path))

    if missing_paths:
        logger.warning(f"Missing data files: {', '.join(missing_paths)}")

    # 验证特征存在
    if not FEATURES:
        logger.error("No features defined in configuration")

    if not TARGET_FEATURE:
        logger.error("No target feature defined in configuration")

    logger.info("Configuration validation completed")


if __name__ == "__main__":
    validate_config()