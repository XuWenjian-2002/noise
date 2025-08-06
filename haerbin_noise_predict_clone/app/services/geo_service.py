# 地理计算
import functools
import math
from typing import Tuple, Dict, Optional
from geopy.distance import geodesic
import pandas as pd
from pathlib import Path

from root_utils.config import DATA_DIR


class GeoService:

    def __init__(self, site_data_dir: Path = DATA_DIR):
        try:
            self.SITE_DATA_PATH = site_data_dir  / "site_coordinates.csv"
            self.site_coordinates = self.load_site_coordinates()
        except Exception as e:
            print(f"初始化站点坐标失败: {str(e)}")
            self.site_coordinates = {}  # 确保属性存在

    def load_site_coordinates(self) -> Dict[str, Tuple[float, float, float]]:
        """从CSV文件加载监测站点坐标数据"""
        try:
            # 尝试加载CSV文件
            site_df = pd.read_csv(self.SITE_DATA_PATH)
            # 尝试匹配可能的列名
            name_col = '监测站点' if '监测站点' in site_df.columns else '站点名称'
            lat_col = 'SiteLat' if 'SiteLat' in site_df.columns else '纬度'
            lon_col = 'SiteLon' if 'SiteLon' in site_df.columns else '经度'
            alt_col = 'SiteAltitude' if 'SiteAltitude' in site_df.columns else '海拔'
            return {
                row[name_col]: (row[lat_col], row[lon_col], row[alt_col])
                for _, row in site_df.iterrows()
            }
        except Exception as e:
            # 文件不存在时返回默认站点数据
            print(f"警告：站点坐标文件 {self.SITE_DATA_PATH} 未找到，使用默认数据")
            return {
                'Site1': (31.2304, 121.4737, 10.5),
                'Site2': (31.2356, 121.4789, 12.3)
            }

    def get_available_sites(self) -> list:
        """获取可用的监测站点列表"""
        return list(self.site_coordinates.keys())

    def calculate_distance(self, aircraft_lat: float, aircraft_lon: float, aircraft_alt: float,
                           site_name: str) -> Optional[float]:
        """
        计算飞机与监测站点的三维距离
        :param aircraft_lat: 飞机纬度
        :param aircraft_lon: 飞机经度
        :param aircraft_alt: 飞机高度(米)
        :param site_name: 监测站点名称
        :return: 三维距离(米)，如果站点不存在返回None
        """
        if site_name not in self.site_coordinates:
            print(f"错误: 未知监测站点 {site_name}")
            return None

        site_lat, site_lon, site_alt = self.site_coordinates[site_name]

        @functools.lru_cache(maxsize=128)
        def _cached_calculation(lat1, lon1, alt1, lat2, lon2, alt2):
            horizontal = geodesic((lat1, lon1), (lat2, lon2)).meters
            vertical = abs(alt1 - alt2)
            return math.sqrt(horizontal ** 2 + vertical ** 2)

        return _cached_calculation(
            aircraft_lat, aircraft_lon, aircraft_alt,
            site_lat, site_lon, site_alt
        )

    def add_site(self, site_name: str, lat: float, lon: float, alt: float):
        """添加新的监测站点（追加模式）"""
        # 添加前检查站点是否已存在
        if site_name in self.site_coordinates:
            print(f"警告: 站点 {site_name} 已存在，将更新坐标")

        # 更新内存中的坐标数据
        self.site_coordinates[site_name] = (lat, lon, alt)

        # 直接追加数据到CSV文件
        self.save_site_coordinates(new_site=site_name)

    def save_site_coordinates(self, new_site: str = None):
        """保存监测站点坐标到CSV文件（支持追加模式）"""
        # 确保目录存在
        self.SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 处理新添加的站点（追加模式）
        if new_site and self.SITE_DATA_PATH.exists():
            # 只处理新添加的站点
            lat, lon, alt = self.site_coordinates[new_site]
            new_row = pd.DataFrame([{
                '监测站点': new_site,
                'SiteLat': lat,
                'SiteLon': lon,
                'SiteAltitude': alt
            }])

            # 追加新行到现有文件
            new_row.to_csv(
                self.SITE_DATA_PATH,
                mode='a',
                header=False,
                index=False
            )
            return

        # 完整写入或覆盖模式
        site_df = pd.DataFrame([
            {'监测站点': name, 'SiteLat': lat, 'SiteLon': lon, 'SiteAltitude': alt}
            for name, (lat, lon, alt) in self.site_coordinates.items()
        ])
        site_df.to_csv(self.SITE_DATA_PATH, index=False)