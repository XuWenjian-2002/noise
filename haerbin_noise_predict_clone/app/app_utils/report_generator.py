# app_utils/report_generator.py
from jinja2 import Environment, FileSystemLoader
import streamlit as st


def get_noise_recommendations(param):
    pass


class ReportGenerator:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('app/templates'))

    def create_flight_report(self, prediction_data):
        template = self.env.get_template("report.html")
        return template.render(
            aircraft=prediction_data['aircraft'],
            noise_level=prediction_data['noise'],
            impact_level=self.calculate_impact(prediction_data['noise'])
        )

    def create_comparative_report(self, predictions):
        pass
        # 多种机型比较报告...

    def generate_report(prediction_data):
        # 设置模板目录
        env = Environment(loader=FileSystemLoader('app/templates'))

        # 选择报告模板
        template = env.get_template("flight_report.html")

        # 渲染报告
        html_report = template.render(
            aircraft_type=prediction_data['type'],
            noise_level=prediction_data['value'],
            recommendations=get_noise_recommendations(prediction_data['value'])
        )

        # 提供下载
        st.download_button("下载完整报告", html_report, "flight_report.html")