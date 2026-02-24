
# 📊 National Stats - 国家统计局数据分析系统
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
一个功能强大、多模式运行的国家统计局公开数据爬取、分析与预测系统。支持 **GUI桌面应用**、**Web交互仪表盘** 和 **CLI命令行工具** 三种运行模式，包含机器学习预测、自动报告生成及高性能并发爬取功能。
---
## 📑 目录
- [项目特性](#-项目特性)
- [功能演示](#-功能演示)
- [环境要求](#-环境要求)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [核心模块说明](#-核心模块说明)
- [数据分析指标](#-数据分析指标)
- [配置说明](#-配置说明)
- [技术栈](#-技术栈)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)
- [联系方式](#-联系方式)
---
## 🚀 项目特性
### 🛠 多种运行模式
- **🖥️ GUI 桌面应用**：基于 PyQt5 构建的现代化桌面客户端，适合日常交互使用。
- **🌐 Web 仪表盘**：基于 Streamlit 构建的交互式网页应用，支持实时交互和远程访问。
- **⌨️ CLI 命令行工具**：支持批处理、自动化任务和脚本调用，适合开发者和运维人员。
### 📈 数据能力
- **多维度数据源**：涵盖 GDP、人口、CPI、贸易、工业、投资、零售、收入等 8 大核心指标。
- **高性能爬取**：支持多线程并发爬取，内置本地缓存机制（支持过期时间设置），大幅提升效率。
- **智能预测**：集成线性回归、多项式回归、指数增长模型及 ARIMA 时间序列分析，预测未来趋势。
### 📊 可视化与报告
- **专业图表**：自动生成折线图、柱状图、堆叠图、热力图、雷达图等。
- **自动报告**：一键生成包含数据分析、预测结果、图表的 Markdown 和 PDF 专业报告。
- **多格式导出**：支持导出为 Excel (.xlsx), CSV, JSON 格式。
---
## 🖼 功能演示
*(此处可放置项目截图，建议在 GitHub 上传图片)*
> **GUI 界面预览**：数据概览、趋势分析、预测图表。
> **Web 仪表盘预览**：交互式 Plotly 图表，实时参数调整。
---
## 💻 环境要求
- Python 3.8 或更高版本
- 操作系统：Windows / macOS / Linux
---
## 🔧 安装指南
### 1. 克隆项目
```bash
git clone https://github.com/yourusername/national_stats.git
cd national_stats
```
### 2. 创建虚拟环境 (推荐)
```bash
# Windows
python -m venv venv
venv\Scripts\activate
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. 安装依赖
```bash
pip install -r requirements.txt
```
### 4. 安装中文字体 (可选，用于 PDF 报告生成)
下载 `simhei.ttf` 字体文件并放置在项目根目录下，以确保 PDF 报告中的中文正常显示。
---
## 🏃 快速开始
### 模式一：桌面 GUI 应用
适合非技术用户，提供完整的图形化交互界面。
```bash
python main_gui.py
```
### 模式二：Web 仪表盘
适合数据演示和交互式探索，支持浏览器访问。
```bash
streamlit run web_app.py
```
访问 `http://localhost:8501` 查看仪表盘。
### 模式三：命令行工具 (CLI)
适合自动化脚本或批量数据处理。
```bash
# 获取所有数据并导出 Excel
python cli.py fetch --metric all --start 2010 --end 2024 --export excel
# 分析数据并生成预测报告
python cli.py analyze --all --report
# 预测未来 5 年 GDP
python cli.py predict --metric gdp --years 5
```
---
## 📁 项目结构
```
national_stats/
├── config.py              # 全局配置（路径、爬虫参数、指标定义）
├── main_gui.py            # GUI 启动入口
├── web_app.py             # Web 仪表盘启动入口
├── cli.py                 # 命令行工具入口
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
│
├── core/                  # 核心功能模块
│   ├── scraper.py         # 爬虫模块 (并发、缓存、请求封装)
│   ├── analyzer.py        # 分析模块 (统计指标、相关性分析)
│   ├── predictor.py       # 预测模块 (机器学习预测模型)
│   └── reporter.py        # 报告生成器 (Markdown, PDF)
│
├── ui/                    # 用户界面模块
│   └── main_window.py     # PyQt5 主窗口逻辑
│
└── data/                  # 数据存储目录 (自动生成)
    ├── cache/             # 本地缓存
    ├── exports/           # 导出文件
    └── charts/            # 图表图片
```
---
## 📦 核心模块说明
### 1. `core/scraper.py` - 高性能爬虫
- 使用 `ThreadPoolExecutor` 实现并发爬取，速度提升 N 倍。
- 内置 `CacheManager`，支持缓存过期检测，避免重复请求。
- 包含完整的请求重试机制和异常处理。
### 2. `core/predictor.py` - 机器学习预测
- **线性回归**：适用于短期平稳趋势。
- **多项式回归**：捕捉非线性变化拐点。
- **指数增长模型**：适用于 GDP 等长期增长指标。
- 自动评估 R² 分数，选择最佳拟合模型。
### 3. `core/reporter.py` - 报告生成
- 自动生成带图表的 Markdown 报告。
- 支持生成 PDF 格式分析报告（需字体支持）。
---
## 📊 数据分析指标
| 指标代码 | 名称 | 单位 | 说明 |
| :--- | :--- | :--- | :--- |
| `gdp` | 国内生产总值 | 亿元 | 包含三大产业增加值 |
| `population` | 人口数据 | 万人 | 包含城镇化率、出生率等 |
| `cpi` | 居民消费价格指数 | % | 同比涨跌幅 |
| `trade` | 进出口贸易 | 亿美元 | 出口、进口及顺差 |
| `industry` | 工业增加值 | % | 增长率数据 |
| `investment` | 固定资产投资 | 亿元 | - |
| `retail` | 社会消费品零售总额 | 亿元 | - |
| `income` | 居民人均可支配收入 | 元 | - |
---
## ⚙️ 配置说明
编辑 `config.py` 可修改以下参数：
```python
# 爬虫配置
SCRAPER_CONFIG = {
    'timeout': 30,           # 请求超时时间
    'retry_times': 3,        # 重试次数
    'delay': 0.5,            # 请求间隔
    'concurrent_limit': 5,   # 最大并发数
    'cache_expire_hours': 24 # 缓存有效期
}
```
---
## 🧰 技术栈
- **语言**: Python 3.8+
- **GUI**: PyQt5
- **Web**: Streamlit, Plotly
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **机器学习**: Scikit-learn, SciPy, Statsmodels
- **爬虫**: Requests, BeautifulSoup
---
## 🤝 贡献指南
欢迎提交 Issue 和 Pull Request。
1. Fork 本仓库。
2. 创建新分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)。
4. 推送到分支 (`git push origin feature/AmazingFeature`)。
5. 提交 Pull Request。
---
## 📄 许可证
本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。
---
## 📧 联系方式
如有任何问题或建议，欢迎联系：
**Email**: [x18825407105@163.com](mailto:x18825407105@163.com)
项目地址: [https://github.com/yourusername/national_stats](https://github.com/yourusername/national_stats)
---
> ⚠️ **免责声明**: 本项目仅供学习和研究使用，数据版权归国家统计局所有。请勿用于商业用途，爬取频率请遵守网站 robots.txt 协议。
```
