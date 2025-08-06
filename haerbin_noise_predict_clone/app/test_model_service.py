import shutil
import sys
from pathlib import Path

from app_utils.config import BASE_DIR, DATA_DIR

sys.path.append(str(BASE_DIR))

import pytest
import numpy as np
from app.services.model_service import ModelService
from app.services.geo_service import GeoService
from app.app_utils.formatters import DataFormatter
from app.app_utils.validators import ValidationService


@pytest.fixture
def model_service():
    return ModelService()


@pytest.fixture
def geo_service():
    return GeoService()


def test_predict(model_service):
    aircraft_type = 'A320'
    input_data = {
        'dateTime': '2025-03-01 00:06:16',
        'Course': 219.126036,
        'Speed': 319.903182,
        'Altitude': 451.7,
        'WindSpeed': 0.3,
        'AirTemperature': -26,
        'Distance': 1506.3257768350024,
        'RelativeHumidity': 56.4
    }
    try:
        prediction = model_service.predict(aircraft_type, input_data)
        assert isinstance(prediction, (float, np.float32, np.float64))
    except ValueError as e:
        assert str(e) == f"不支持机型: {aircraft_type}"


def test_load_scalers(model_service):
    """测试加载特定机型的缩放器"""
    aircraft_type = 'A320'  # 使用截图中的测试机型

    # 测试加载
    model_service.load_scalers(aircraft_type)

    # 验证特征缩放器加载
    assert aircraft_type in model_service.feature_scalers
    assert model_service.feature_scalers[aircraft_type] is not None

    # 验证目标缩放器加载
    assert aircraft_type in model_service.target_scalers
    assert model_service.target_scalers[aircraft_type] is not None


def test_load_model(model_service):
    model_service.load_model('A320')
    assert 'A320' in model_service.models


def test_prepare_features(model_service):
    input_data = {
        'dateTime': '2023-01-01 12:00:00',
        'Course': 180,
        'Speed': 800,
        'Altitude': 3000,
        'WindSpeed': 5.0,
        'AirTemperature': 25.0,
        'Distance': 1500,
        'RelativeHumidity': 50
    }
    features = model_service._prepare_features(input_data)
    assert isinstance(features, np.ndarray)


@pytest.fixture
def geo_service(tmp_path):
    # 使用临时目录替代真实文件
    test_csv = tmp_path / "test_sites.csv"
    # 复制真实数据到临时文件（可选）
    if Path(f"{DATA_DIR} / site_coordinates.csv").exists():
        shutil.copy("real_data_path.csv", test_csv)

    # 创建测试专用的 GeoService 实例
    service = GeoService(site_data_path=test_csv)
    yield service
    # 无需清理，tmp_path 会自动回收

def test_geo_service_load_site_coordinates(geo_service):
    site_coordinates = geo_service.load_site_coordinates()
    assert isinstance(site_coordinates, dict)


def test_geo_service_get_available_sites(geo_service):
    sites = geo_service.get_available_sites()
    assert isinstance(sites, list)


def test_geo_service_calculate_distance(geo_service):
    geo_service.load_site_coordinates()  # 确保加载最新数据
    aircraft_lat = 45.656
    aircraft_lon = 126.259
    aircraft_alt = 161
    site_name = '徐家窝棚'
    distance = geo_service.calculate_distance(aircraft_lat, aircraft_lon, aircraft_alt, site_name)
    assert isinstance(distance, float) or distance is None


def test_geo_service_add_site(geo_service):
    site_name = '什么琪吃大便'
    lat = 66.666
    lon = 188.1888
    alt = 15.8
    geo_service.add_site(site_name, lat, lon, alt)
    assert site_name in geo_service.site_coordinates


def test_data_formatter_format_prediction_input():
    input_data = {
        'Course': 180,
        'Speed': 800,
        'Altitude': 3000,
        'WindSpeed': 5.0,
        'AirTemperature': 25.0
    }
    formatted = DataFormatter.format_prediction_input(input_data)
    assert 'dateTime' in formatted
    assert 'Aircraft_model' in formatted


def test_data_formatter_format_prediction_result():
    prediction = 80.0
    result = DataFormatter.format_prediction_result(prediction)
    assert 'noise_level' in result
    assert 'classification' in result


def test_validation_service_validate_coordinates():
    lat = 31.2304
    lon = 121.4737
    valid = ValidationService.validate_coordinates(lat, lon)
    assert isinstance(valid, bool)


def test_validation_service_validate_aircraft_type():
    aircraft_type = 'A320'
    allowed_types = ['A320', 'B738']
    valid = ValidationService.validate_aircraft_type(aircraft_type, allowed_types)
    assert isinstance(valid, bool)


def test_validation_service_validate_environment_params():
    temp = 25.0
    humidity = 60
    wind = 5.0
    valid = ValidationService.validate_environment_params(temp, humidity, wind)
    assert isinstance(valid, bool)


def test_validation_service_validate_all_inputs():
    inputs = {
        'Lat': 31.2304,
        'Lon': 121.4737,
        'AirTemperature': 25.0,
        'RelativeHumidity': 60,
        'WindSpeed': 5.0
    }
    valid = ValidationService.validate_all_inputs(inputs)
    assert isinstance(valid, bool)