# 输入验证

class ValidationService:
    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """验证经纬度是否在有效范围内"""
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def validate_aircraft_type(aircraft_type: str, allowed_types: list) -> bool:
        """验证机型是否在允许列表中"""
        return aircraft_type in allowed_types

    @staticmethod
    def validate_environment_params(temp: float, humidity: int, wind: float) -> bool:
        """验证环境参数是否合理"""
        return (
            -50 <= temp <= 50 and
            0 <= humidity <= 100 and
            0 <= wind <= 50
        )

    @classmethod
    def validate_all_inputs(cls, inputs: dict) -> bool:
        """综合验证所有输入参数"""
        valid_coords = cls.validate_coordinates(inputs['Lat'], inputs['Lon'])
        valid_env = cls.validate_environment_params(
            inputs['AirTemperature'],
            inputs['RelativeHumidity'],
            inputs['WindSpeed']
        )
        return valid_coords and valid_env