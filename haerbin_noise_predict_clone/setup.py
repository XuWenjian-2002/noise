import subprocess
import sys

from root_utils.config import BASE_DIR


def main():
    # 获取项目根目录

    print(f"项目根目录: {BASE_DIR}")

    # 应用主脚本路径
    main_script = BASE_DIR / "app" / "main.py"
    print(f"应用入口: {main_script}")

    if not main_script.exists():
        print(f"❌ 错误: 找不到应用入口文件 {main_script}")
        sys.exit(1)

    print("🚀 启动飞机噪声预测系统...")

    # 尝试使用随机端口启动（端口0表示由系统自动分配）
    print("尝试使用随机端口启动...")
    subprocess.run(
        ["streamlit", "run", str(main_script)],
        check=True,
        cwd=BASE_DIR
    )



if __name__ == "__main__":
    main()