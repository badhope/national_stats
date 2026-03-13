# 宏观经济数据分析平台

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8+-green" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
</p>

专业的宏观经济数据分析和可视化平台，基于国家统计局官方数据，提供全面的经济指标分析、趋势预测和报告生成服务。

## 📊 项目简介

宏观经济数据分析平台是一个专注于中国宏观经济数据的研究工具，旨在为经济学家、研究人员和政策制定者提供高效、准确的数据分析服务。平台整合了 GDP、CPI、就业、贸易等核心经济指标，支持多维度数据可视化和智能趋势预测。

## ✨ 功能特点

### 核心功能

| 功能 | 说明 |
|------|------|
| 📈 多维数据可视化 | 支持折线图、柱状图、饼图、热力图等多种可视化形式 |
| 📥 一键数据导出 | 支持 Excel、CSV、JSON、PDF 等多种格式导出 |
| 🔮 智能趋势预测 | 基于 ARIMA、机器学习等算法提供趋势预测 |
| 🔒 数据安全保障 | 官方数据源授权，传输加密存储 |

### 数据指标

- **GDP**: 国内生产总值及增长率
- **CPI**: 居民消费价格指数
- **就业**: 城镇失业率、就业人数
- **贸易**: 进出口总额、贸易顺差

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- Windows / macOS / Linux

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/badhope/national_stats.git
cd national_stats

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行程序

```bash
# 运行主程序
python app/main.py

# 查看帮助
python app/main.py --help

# 仅加载数据
python app/main.py --data-only

# 仅执行分析
python app/main.py --analyze-only
```

### 查看产品介绍网页

```bash
# 方式1：直接打开
# 双击 docs-presentation/index.html

# 方式2：本地服务器
cd docs-presentation
python -m http.server 8080
# 访问 http://localhost:8080
```

## 📁 项目结构

```
national_stats/
├── app/                        # 功能程序模块
│   ├── main.py                 # 主程序入口
│   ├── requirements.txt        # Python依赖
│   └── README.txt             # 程序说明
│
├── docs-presentation/          # 产品介绍网页
│   └── index.html             # 产品首页
│
├── core/                       # 核心模块库
│   ├── analyzer.py            # 数据分析器
│   ├── data_manager.py        # 数据管理器
│   ├── predictor.py           # 趋势预测
│   ├── visualizer.py         # 数据可视化
│   ├── reporter.py           # 报告生成
│   ├── database.py           # 数据库模块
│   ├── cache.py              # 缓存管理
│   ├── data_sources/         # 数据源
│   │   ├── nbs.py           # 国家统计局数据
│   │   └── mock.py          # 模拟数据
│   └── models/               # 经济模型
│       ├── gdp_model.py
│       └── ...
│
├── cli.py                     # 命令行工具
├── config.py                 # 配置文件
├── start.py                  # 启动脚本
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明
├── LICENSE                   # 许可证
└── .gitignore               # Git忽略配置
```

## 📖 使用文档

### 命令行工具

```bash
# 启动交互式界面
python start.py

# 使用CLI工具
python cli.py --help
python cli.py analyze --type gdp
python cli.py visualize --type gdp --format png
python cli.py predict --type gdp --years 5
```

### 程序输出

运行程序后会在以下目录生成输出：

```
output/
├── analysis/    # 分析结果
├── charts/      # 可视化图表
├── reports/     # 生成报告
└── data/        # 导出的数据

logs/            # 运行日志
cache/           # 缓存数据
```

## 🔧 配置说明

编辑 `config.py` 自定义配置：

```python
# 数据源配置
DATA_SOURCE = "nbs"  # 或 "mock"

# 输出目录
OUTPUT_DIR = "output"

# 可视化配置
CHART_DPI = 300
CHART_THEME = "seaborn"

# 预测配置
PREDICTION_YEARS = 5
```

## 🛠️ 技术栈

- **编程语言**: Python 3.8+
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn, plotly
- **统计分析**: scipy, statsmodels
- **机器学习**: scikit-learn
- **Web框架**: Flask (可选)
- **前端**: HTML5, CSS3, JavaScript

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📝 更新日志

### v1.0.0 (2024-01)
- ✅ 初始版本发布
- ✅ GDP/CPI/就业数据分析
- ✅ 趋势预测功能
- ✅ 产品介绍网页

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- 项目主页: https://github.com/badhope/national_stats
- 问题反馈: https://github.com/badhope/national_stats/issues

---

<p align="center">Made with ❤️ by National Stats Team</p>
