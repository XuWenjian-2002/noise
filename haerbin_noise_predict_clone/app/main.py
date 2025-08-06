# Streamlit主应用
import sys
from pathlib import Path
# 获取项目根目录路径
BASE_DIR = Path(__file__).resolve().parent.parent
# 将项目根目录添加到系统路径
sys.path.insert(0,str(BASE_DIR))

import chardet
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import shap
import matplotlib.pyplot as plt

try:
    # 使用完整路径导入根目录的utils
    from root_utils.config import MODEL_SAVE_DIR, DATA_DIR, STATIC_DIR, APP_CONFIG
except ImportError:
    # 回退到原始导入方式（仅用于本地测试）
    from utils.config import MODEL_SAVE_DIR, DATA_DIR, STATIC_DIR, APP_CONFIG

# 导入app/utils的内容保持不变
from app_utils.formatters import DataFormatter
from app_utils.validators import ValidationService




# 初始化服务(带缓存)
@st.cache_resource(ttl=3600, show_spinner="正在初始化服务...")
def init_services():
    # 延迟导入只在需要时加载
    from services.model_service import ModelService
    from services.geo_service import GeoService

    # 不需要传递load_models参数
    model_service = ModelService()
    geo_service = GeoService()

    import logging
    import os
    logging.info(f"模型目录: {os.path.abspath(model_service.models_dir)}")
    logging.info(f"目录内容: {os.listdir(model_service.models_dir)}")

    return model_service, geo_service

model_service, geo_service = init_services()

# 应用配置
st.set_page_config(
    page_title="飞机噪声预测系统",
    page_icon="✈️",
    layout="wide"
)


# 自定义CSS
@st.cache_resource
def load_css():
    css_file = STATIC_DIR / "css" / "styles.css"
    if css_file.exists():
        try:
            with open(css_file, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
            return raw_data.decode(encoding or 'utf-8', errors='replace')
        except UnicodeDecodeError:
            # 回退到latin-1编码（不会抛出解码错误）
            with open(css_file, encoding='latin-1', errors='ignore') as f:
                return f.read()
    return ""

st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


# 添加可视化组件
def create_noise_visualization(prediction: float):
    """创建噪声水平可视化图表,延迟加载可视化库"""
    # 需要时再导入
    import plotly.express as px

    # 使用更高效的Plotly替代Altair
    levels = ['Low', 'Moderate', 'High']
    values = [min(prediction, 70.0), max(0.0, min(prediction - 70, 15)), max(0.0, prediction - 85)]

    fig = px.bar(
        x=values,
        y=levels,
        orientation='h',
        color=levels,
        color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'},
        labels={'x': '噪声值 (dB)', 'y': ''}
    )
    fig.update_layout(showlegend=False, height=300)
    return fig

# 地图显示函数
def display_map(lon, lat, site_name, distance):
    """优化地图渲染性能"""
    # 使用预加载的坐标数据
    site_coords = geo_service.site_coordinates.get(site_name)
    if not site_coords:
        return None

    site_lat, site_lon, _ = site_coords

    # 使用更轻量级的初始视图
    m = folium.Map(
        location=[(lat + site_lat) / 2, (lon + site_lon) / 2],
        zoom_start=APP_CONFIG["map_default_zoom"],
        tiles='CartoDB positron'  # 更轻量级的瓦片
    )

    # 简化标记
    folium.Marker([lat, lon], popup=f"飞机位置\n高度: {Altitude}m").add_to(m)
    folium.Marker([site_lat, site_lon], popup=f"{site_name}\n距离: {distance:.0f}m").add_to(m)

    # 使用简化线条
    folium.PolyLine([[lat, lon], [site_lat, site_lon]], color='blue', weight=2).add_to(m)

    return folium_static(m, width=800, height=500)


# 历史数据加载函数（带缓存）
@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def load_history_data(site_name: str, date_range: tuple) -> pd.DataFrame:
    """带缓存的懒加载历史数据

    参数:
        site_name: 监测站点名称
        start_date: 开始日期
        end_date: 结束日期

    返回:
        pd.DataFrame: 历史数据DataFrame
    """
    """优化参数传递和缓存键"""
    start_date, end_date = date_range

    # 使用更有效的缓存键
    cache_key = f"{site_name}-{start_date:%Y%m%d}-{end_date:%Y%m%d}"
    # 如果缓存中已有数据，直接返回
    if cache_key in load_history_data._cache:
        return load_history_data._cache[cache_key]

    # 使用更高效的文件读取方式
    file_path = DATA_DIR / f"{site_name}_history.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path,
                             columns=['timestamp', 'noise_level'],
                             filters=[('timestamp', '>=', start_date),('timestamp', '<=', end_date)])
    else:
        # 回退到CSV
        csv_path = DATA_DIR / f"{site_name}_history.csv"
        df = pd.read_csv(csv_path,
                         parse_dates=['timestamp'],
                         usecols=['timestamp', 'noise_level'])
        df.to_parquet(file_path)  # 转换为parquet格式加快后续加载

    # 使用更高效的日期过滤
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    return df.loc[mask].copy()  # 返回副本避免修改缓存数据


# 初始化缓存字典
if not hasattr(load_history_data, '_cache'):
    load_history_data._cache = {}


# 轨迹数据加载函数（带缓存）
@st.cache_data(ttl=600, show_spinner="加载轨迹数据...")  # 缓存10分钟
def load_flight_trajectory(aircraft_type: str) -> pd.DataFrame:
    """带缓存的懒加载飞行轨迹数据

    参数:
        aircraft_type: 飞机类型

    返回:
        pd.DataFrame: 飞行轨迹数据
    """
    # 生成缓存键
    cache_key = f"trajectory-{aircraft_type}"

    # 如果缓存中已有数据，直接返回
    if cache_key in load_flight_trajectory._cache:
        return load_flight_trajectory._cache[cache_key]

    # 否则从数据源加载
    try:
        # 实际数据加载逻辑 - 这里替换为您的真实数据加载代码
        file_path = DATA_DIR / f"{aircraft_type}_trajectory.csv"
        df = pd.read_csv(file_path,
                         parse_dates=['timestamp'],
                         usecols=['timestamp', 'lat', 'lon', 'altitude', 'distance', 'speed'])

        # 缓存结果
        load_flight_trajectory._cache[cache_key] = df
        return df
    except Exception as e:
        st.error(f"加载飞行轨迹失败: {str(e)}")
        return pd.DataFrame()


def explain_prediction(input_data: dict, model):
    """
    使用SHAP值解释模型预测结果

    参数:
        input_data: 预测时使用的输入数据
        model: 使用的预测模型
    """
    try:
        # 将输入数据转换为DataFrame
        input_df = pd.DataFrame([input_data])

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)

        # 计算SHAP值
        shap_values = explainer.shap_values(input_df)

        # 可视化解释
        st.subheader("特征影响分析")

        # 创建水平条形图展示特征重要性
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)

        # 显示详细解释
        st.subheader("详细特征贡献")
        for i, feature in enumerate(input_df.columns):
            contribution = shap_values[0][i]
            st.write(f"- **{feature}**: {input_data[feature]} → 贡献值: {contribution:.4f}")

        # 显示基础值
        base_value = explainer.expected_value
        st.caption(f"预测基准值: {base_value:.4f}")

    except Exception as e:
        st.error(f"解释失败: {str(e)}")
        st.exception(e)


# 初始化缓存字典
if not hasattr(load_flight_trajectory, '_cache'):
    load_flight_trajectory._cache = {}

# 侧边栏输入
with st.sidebar:
    st.title("参数设置")

    aircraft_type = st.selectbox(
        "选择机型",
        options=APP_CONFIG["allowed_aircraft_types"],
        index=0
    )

    # 在侧边栏的预加载按钮处
    if st.button("预加载模型", help="提前加载选中的机型模型"):
        if aircraft_type:
            with st.spinner(f"正在加载{aircraft_type}模型..."):
                try:
                    model_service.load_model(aircraft_type)
                    st.success(f"{aircraft_type}模型加载完成！")
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
        else:
            st.warning("请先选择要加载的机型!")

    site_name = st.selectbox(
        "监测站点",
        options=list(geo_service.site_coordinates.keys()),
        index=list(geo_service.site_coordinates.keys()).index(APP_CONFIG["default_site"])
    )

    with st.expander("飞行参数"):
        # 添加实际需要的输入项
        Lon = st.number_input("经度", min_value=0.0, max_value=180.0, value=126.201291, format="%.6f")
        Lat = st.number_input("纬度", min_value=0.0, max_value=90.0, value=45.5711977, format="%.6f")
        Altitude = st.number_input("高度(m)", min_value=0.0, value=451.7, format="%.1f")
        Speed = st.number_input("速度(km/h)", min_value=0.0, value=319.903182, format="%.6f")
        Course = st.number_input("航向角(°)", min_value=0.0, max_value=360.0, value=219.126036, format="%.6f")

    with st.expander("气象条件"):
        AirTemperature = st.number_input("温度(℃)", value=-2.6, format="%.1f")
        RelativeHumidity = st.number_input("湿度(%)", min_value=0.0, max_value=100.0, value=56.4, format="%.1f")
        WindSpeed = st.number_input("风速(m/s)", min_value=0.0, value=0.3, format="%.1f")

    # 验证用户输入
    inputs = {
        'Lat': Lat,
        'Lon': Lon,
        'AirTemperature': AirTemperature,
        'RelativeHumidity': RelativeHumidity,
        'WindSpeed': WindSpeed
    }
    if not ValidationService.validate_all_inputs(inputs):
        st.warning("输入参数不合法，请检查！")

    # 格式化输入数据
    input_data = {
        'Course': Course,
        'Speed': Speed,
        'Altitude': Altitude,
        'WindSpeed': WindSpeed,
        'AirTemperature': AirTemperature,
        'Distance': geo_service.calculate_distance(Lat, Lon, Altitude, site_name),
        'RelativeHumidity': RelativeHumidity
    }
    formatted_input_data = DataFormatter.format_prediction_input(input_data)

    # 添加资源清理功能
    st.divider()
    if st.button("清理缓存", help="清除所有缓存数据释放内存"):
        st.cache_data.clear()
        st.cache_resource.clear()
        # 清除自定义缓存
        load_history_data._cache = {}
        load_flight_trajectory._cache = {}
        st.success("缓存已清除! 内存已释放")

# 主界面
st.title("✈️ 飞机噪声预测系统")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["实时预测", "历史分析", "模型管理", "实时轨迹管理", "预测结果解释", "站点管理"])

with tab1: # 实时预测
    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("测试预测"):
            test_aircraft_type = 'A320'
            test_input_data = {
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
                prediction = model_service.predict(test_aircraft_type, test_input_data)
                st.success(f"测试预测结果: {prediction:.2f} dB")
            except Exception as e:
                st.error(f"测试预测失败: {str(e)}")
        # 地图容器
        map_placeholder = st.empty()

    with col2:
        # 输入参数摘要
        st.subheader("当前参数")
        st.metric("机型", aircraft_type)
        st.metric("站点", site_name)

        # 计算并显示距离
        try:
            Distance = geo_service.calculate_distance(Lat, Lon, Altitude, site_name)
            st.metric("距离", f"{Distance:.0f}m")
        except Exception as e:
            st.warning(f"距离计算失败: {str(e)}")
            Distance = None

        # 预测按钮和结果
        if st.button("执行预测", type="primary", use_container_width=True):
            if Distance is not None:
                with st.spinner("计算中..."):
                    try:
                        # 准备输入数据
                        input_data = {
                            'Distance': Distance,
                            'Altitude': Altitude,
                            'Speed': Speed,
                            'AirTemperature': AirTemperature,
                            'RelativeHumidity': RelativeHumidity,
                            'WindSpeed': WindSpeed
                        }

                        # 执行预测
                        prediction = model_service.predict(aircraft_type, input_data)

                        # 显示可视化
                        st.subheader("噪声水平")
                        formatted_result = DataFormatter.format_prediction_result(prediction)
                        st.success(f"预测噪声值: {formatted_result['noise_level']}")
                        st.write(f"分类: {formatted_result['classification']}")
                        st.altair_chart(create_noise_visualization(prediction))

                        # 更新地图
                        with map_placeholder:
                            display_map(Lon, Lat, site_name, Distance)

                        # 存储预测结果到会话状态
                        st.session_state['last_prediction'] = {
                            'input': input_data,
                            'output': prediction,
                            'model': aircraft_type
                        }

                    except Exception as e:
                        st.error(f"预测失败: {str(e)}")
            else:
                st.warning("无法执行预测，请先解决距离计算问题")

with tab2: # 历史分析
    st.header("历史数据分析")
    # 选择站点和日期范围
    col1, col2 = st.columns(2)
    with col1:
        site_name_hist = st.selectbox("监测站点",
                                     options=list(geo_service.site_coordinates.keys()),
                                     key="hist_site")
    with col2:
        date_range = st.date_input("选择日期范围", [], key="hist_dates")

    if st.button("加载历史数据"):
        if len(date_range) == 2:
            # 使用懒加载函数获取数据
            history_df = load_history_data(site_name_hist, date_range[0], date_range[1])

            if not history_df.empty:
                st.line_chart(history_df.set_index('timestamp')['noise_level'])

                # 显示统计信息
                st.subheader("统计摘要")
                st.write(f"平均噪声: {history_df['noise_level'].mean():.2f} dB")
                st.write(f"最大值: {history_df['noise_level'].max():.2f} dB")
                st.write(f"最小值: {history_df['noise_level'].min():.2f} dB")
            else:
                st.warning("该时间段无可用数据")
        else:
            st.warning("请选择完整的日期范围")

with tab3: # 模型管理
    st.header("模型管理")
    uploaded_model = st.file_uploader("上传新模型", type=['h5', 'pkl'])

    if uploaded_model:
        with open(MODEL_SAVE_DIR / uploaded_model.name, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        st.success(f"模型 {uploaded_model.name} 上传成功！")
        model_service.load_model(aircraft_type)  # 重新加载模型

    st.subheader("现有模型")
    for model_name, model_path in model_service.models.items():
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{model_name}**")
        col2.button("删除", key=model_name, disabled=True)


with tab4:  # 实时轨迹管理
    st.header("飞行轨迹模拟")

    # 选择飞机类型
    aircraft_type_traj = st.selectbox("机型",
                                      options=list(model_service.models.keys()),
                                      key="traj_aircraft")

    if st.button("生成轨迹"):
        # 使用懒加载函数获取轨迹数据
        trajectory_df = load_flight_trajectory(aircraft_type_traj)

        if not trajectory_df.empty:
            # 创建轨迹地图
            m = folium.Map(location=[trajectory_df['lat'].mean(), trajectory_df['lon'].mean()],
                           zoom_start=12)

            # 添加轨迹线
            folium.PolyLine(
                locations=trajectory_df[['lat', 'lon']].values,
                color='blue',
                weight=2.5,
                opacity=0.7
            ).add_to(m)

            # 添加起点和终点标记
            folium.Marker(
                trajectory_df.iloc[0][['lat', 'lon']].values,
                popup="起点",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)

            folium.Marker(
                trajectory_df.iloc[-1][['lat', 'lon']].values,
                popup="终点",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

            folium_static(m)

            # 显示轨迹统计信息
            st.subheader("轨迹统计")
            col1, col2, col3 = st.columns(3)
            col1.metric("总距离", f"{trajectory_df['distance'].sum():.1f} km")
            col2.metric("平均高度", f"{trajectory_df['altitude'].mean():.0f} m")
            col3.metric("平均速度", f"{trajectory_df['speed'].mean():.0f} km/h")
        else:
            st.warning("无可用轨迹数据")

with tab5: # 预测结果解释功能
    if 'prediction' in st.session_state:
        st.header("预测结果解释")
        explain_prediction(st.session_state['input_data'], model_service.load_model(aircraft_type))
    else:
        st.info("请先执行预测以查看解释")

with tab6: # 站点管理
    with st.expander("站点管理"):
        new_site = st.text_input("新站点名称")
        new_lat = st.number_input("纬度")
        new_lon = st.number_input("经度")
        new_alt = st.number_input("海拔")
        if st.button("添加站点"):
            geo_service.add_site(new_site, new_lat, new_lon, new_alt)
