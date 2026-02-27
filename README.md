# 📊 宏观经济智能分析平台 (National Statistics Intelligence Platform)
[![GitHub Stars](https://img.shields.io/github/stars/[你的GitHub用户名]/national_stats?style=flat-square&color=yellow)](https://github.com/[你的GitHub用户名]/national_stats/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/[你的GitHub用户名]/national_stats?style=flat-square&color=blue)](https://github.com/[你的GitHub用户名]/national_stats/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/[你的GitHub用户名]/national_stats?style=flat-square&color=orange)](https://github.com/[你的GitHub用户名]/national_stats/issues)
[![GitHub License](https://img.shields.io/github/license/[你的GitHub用户名]/national_stats?style=flat-square&color=green)](https://github.com/[你的GitHub用户名]/national_stats/blob/main/LICENSE)
[![Language](https://img.shields.io/badge/language-Python%203.9+-purple?style=flat-square)](https://github.com/[你的GitHub用户名]/national_stats)
[![Version](https://img.shields.io/github/v/release/[你的GitHub用户名]/national_stats?style=flat-square)](https://github.com/[你的GitHub用户名]/national_stats/releases)
[![Downloads](https://img.shields.io/github/downloads/[你的GitHub用户名]/national_stats/total?style=flat-square)](https://github.com/[你的GitHub用户名]/national_stats/archive/refs/tags/latest.zip)

一个功能完备的宏观经济数据分析、预测和可视化平台，专为中国宏观经济指标设计。

## 🚀 主要特性

### 🔧 核心功能
- **多源数据获取**: 对接国家统计局、央行、海关等官方数据源
- **智能缓存管理**: Redis + 文件系统双重缓存机制
- **高性能处理**: 支持Dask和Ray分布式计算
- **实时数据更新**: 自动数据同步和增量更新

### 📈 分析能力
- **传统统计分析**: 描述性统计、相关性分析、平稳性检验
- **时间序列预测**: ARIMA、Prophet、XGBoost、集成学习等多种方法
- **高级数据拟合**: 多项式、指数、逻辑斯蒂、样条等多种拟合方法
- **经济模型**: 增长核算、奥肯定律、菲利普斯曲线等经典模型

### 🎯 特色功能
- **大数据洞察**: 批量处理数百个指标，发现隐藏关联模式
- **智能预测**: 自动选择最优预测方法，支持不确定性量化
- **数据拟合**: 强大的曲线拟合能力，支持外推预测
- **异常检测**: 基于统计学的异常值识别
- **聚类分析**: 自动识别指标间的相似性模式

## 📁 项目结构

```
national_stats/
├── LICENSE
├── README.md
├── cli.py
├── config.py
├── module4_cli_enhancer.py
├── requirements.txt
├── start.py
├── web_app_enhanced.py
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── analyzer.py         # 统计分析器
│   ├── big_data_processor.py  # 大数据处理器
│   ├── cache.py            # 缓存管理
│   ├── data_manager.py     # 数据管理器
│   ├── data_sources/       # 数据源管理
│   │   ├── database.py     # 数据库数据源
│   │   └── [补充其他数据源文件]
│   ├── fitter.py           # 数据拟合器
│   ├── models/             # 经济模型
│   ├── predictor.py        # 高级预测器
│   ├── reporter.py         # 报告生成器
│   └── visualizer.py       # 可视化工具
├── models/                 # 数据模型
│   └── time_series.py      # 时间序列模型
```

## 🛠️ 安装与配置

### 环境要求
- Python 3.9+
- 8GB+ 内存推荐
- 现代CPU（支持多核处理）

### 快速安装

```bash
# 克隆项目
git clone https://github.com/[你的GitHub用户名]/national_stats.git
cd national_stats

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装可选的高性能计算库
pip install dask[complete] ray[xgboost]
```

### 配置文件

项目会自动创建必要的目录结构。可根据需要修改 `config.py` 中的配置：

```python
# 自定义配置示例
from config import Config

# 修改数据库配置
Config.database.db_type = "postgresql"
Config.database.connection_string = "postgresql://user:pass@localhost/db"

# 调整性能参数
Config.performance.max_workers = 8
Config.big_data.batch_size = 100
```

## 💻 使用方法

### 1. 命令行界面
```bash
# 基础使用示例
python cli.py --help  # 查看命令帮助
python cli.py analyze --indicator GDP --time-range 2010-2024  # 分析GDP指标
python cli.py predict --indicator CPI --method ARIMA --forecast-period 12  # 预测CPI未来12期数据

# 增强版CLI使用
python module4_cli_enhancer.py --batch-process ./indicators_list.txt  # 批量处理指标列表
```

### 2. Web应用启动
```bash
# 启动基础Web应用（若有）
# python web_app.py

# 启动增强版Web应用
python web_app_enhanced.py --host 0.0.0.0 --port 8080
```

### 3. 快速启动脚本
```bash
python start.py  # 一键启动（集成CLI/Web/数据同步）
```

## 📊 数据可视化与报告
- 支持生成交互式图表（折线图、柱状图、热力图、散点图等）
- 自动生成分析报告（PDF/HTML/Markdown格式）
- 支持自定义报告模板，适配不同业务场景

## 🚀 性能优化
- 分布式计算：基于Dask/Ray实现多节点/多核并行处理
- 缓存策略：Redis缓存高频访问数据，文件系统缓存批量历史数据
- 数据分片：大数据集自动分片处理，降低内存占用

## 🤝 贡献指南
1. Fork 本仓库（https://github.com/[你的GitHub用户名]/national_stats/fork）
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📞 联系方式
- 邮箱：18825407105@outlook.com
- 项目地址：https://github.com/badhope/national_stats
- 问题反馈：https://github.com/badhope/national_stats/issues

## 📄 许可证
本项目基于 [LICENSE](LICENSE) 协议开源。

