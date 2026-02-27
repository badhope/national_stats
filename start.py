#!/usr/bin/env python
"""
项目启动脚本
提供一键启动各种功能的便捷入口
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config


def setup_environment():
    """设置运行环境"""
    print("🔧 正在初始化环境...")
    
    # 确保必要目录存在
    Config.paths.ensure_dirs()
    
    # 检查依赖
    try:
        import pandas
        import numpy
        print("✅ 核心依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少必要依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def start_web_app(mode="enhanced"):
    """启动Web应用"""
    if not setup_environment():
        return
    
    print("🚀 启动Web应用...")
    
    if mode == "enhanced":
        app_file = "web_app_enhanced.py"
        print("使用增强版Web应用 (推荐)")
    else:
        app_file = "web_app.py"
        print("使用基础版Web应用")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Web应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def start_cli():
    """启动命令行界面"""
    if not setup_environment():
        return
    
    print("🖥️  启动命令行界面...")
    
    try:
        # 导入并运行CLI
        from cli import main
        main()
    except KeyboardInterrupt:
        print("\n👋 CLI已退出")
    except Exception as e:
        print(f"❌ CLI启动失败: {e}")


def demo_basic_functions():
    """演示基本功能"""
    if not setup_environment():
        return
    
    print("🧪 演示基本功能...")
    
    try:
        from core import DataManager
        from core.fitter import AdvancedFitter
        
        # 获取示例数据
        print("1. 获取GDP数据...")
        dm = DataManager(use_mock_data=True)  # 使用模拟数据
        gdp_ts = dm.fetch("gdp")
        
        if gdp_ts:
            print(f"   ✅ 成功获取 {len(gdp_ts)} 条GDP数据")
            print(f"   时间范围: {gdp_ts.meta.start_date} 至 {gdp_ts.meta.end_date}")
            print(f"   统计信息: 均值={gdp_ts.mean():.2f}, 标准差={gdp_ts.std():.2f}")
            
            # 数据拟合演示
            print("2. 执行数据拟合...")
            import numpy as np
            x = np.arange(len(gdp_ts.data))
            y = gdp_ts.data['value'].values
            fitter = AdvancedFitter()
            fit_result = fitter.fit(x, y, methods=['polynomial', 'exponential'])
            print(f"   ✅ 拟合完成，最佳方法: {fit_result['best_method']}")
            print(f"   最佳R² = {fit_result['r_squared']:.4f}")
            
            # 简单预测（使用移动平均）
            print("3. 执行简单预测...")
            last_values = gdp_ts.data['value'].tail(3).values
            next_value = np.mean(last_values)  # 简单移动平均预测
            print(f"   ✅ 简单预测下一个值: {next_value:.2f}")
            
        else:
            print("   ❌ 数据获取失败")
            
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")


def check_system_status():
    """检查系统状态"""
    print("📋 系统状态检查...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查核心依赖
    dependencies = [
        ("pandas", "数据分析"),
        ("numpy", "数值计算"),
        ("matplotlib", "可视化"),
        ("streamlit", "Web框架"),
        ("statsmodels", "统计建模"),
        ("sklearn", "机器学习")  # 注意：导入名为sklearn，包名为scikit-learn
    ]
    
    missing_deps = []
    for dep, desc in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep:15} - {desc}")
        except ImportError:
            print(f"❌ {dep:15} - {desc} (缺失)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_deps)}")
        print("请运行: pip install -r requirements.txt")
    else:
        print("\n🎉 所有依赖检查通过！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="宏观经济智能分析平台启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start.py web           # 启动增强版Web应用
  python start.py web --basic   # 启动基础版Web应用
  python start.py cli           # 启动命令行界面
  python start.py demo          # 运行功能演示
  python start.py status        # 检查系统状态
        """
    )
    
    parser.add_argument(
        "command",
        choices=["web", "cli", "demo", "status"],
        help="要执行的命令"
    )
    
    parser.add_argument(
        "--basic",
        action="store_true",
        help="使用基础版Web应用（仅对web命令有效）"
    )
    
    args = parser.parse_args()
    
    if args.command == "web":
        mode = "basic" if args.basic else "enhanced"
        start_web_app(mode)
    elif args.command == "cli":
        start_cli()
    elif args.command == "demo":
        demo_basic_functions()
    elif args.command == "status":
        check_system_status()


if __name__ == "__main__":
    main()