# 📊 宏观经济智能分析平台 (National Statistics Intelligence Platform)
[![GitHub Stars](https://img.shields.io/github/stars/badhope/national_stats?style=flat-square&color=yellow)](https://github.com/badhope/national_stats/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/badhope/national_stats?style=flat-square&color=blue)](https://github.com/badhope/national_stats/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/badhope/national_stats?style=flat-square&color=orange)](https://github.com/badhope/national_stats/issues)
[![GitHub License](https://img.shields.io/github/license/badhope/national_stats?style=flat-square&color=green)](https://github.com/badhope/national_stats/blob/main/LICENSE)
[![Language](https://img.shields.io/badge/language-Python%203.9+-purple?style=flat-square)](https://github.com/badhope/national_stats)
[![Version](https://img.shields.io/github/v/release/badhope/national_stats?style=flat-square)](https://github.com/badhope/national_stats/releases)
[![Downloads](https://img.shields.io/github/downloads/badhope/national_stats/total?style=flat-square)](https://github.com/badhope/national_stats/archive/refs/tags/latest.zip)

一个功能完备的宏观经济数据分析、预测和可视化平台，专为中国宏观经济指标设计。

**项目目标**：为经济研究人员、政策制定者、投资者和企业提供专业、高效的宏观经济数据分析工具，帮助用户快速获取、分析和预测宏观经济走势。

**核心价值**：
- 📊 **数据集成**：整合多源官方数据，提供统一的访问接口
- 📈 **智能分析**：内置多种统计分析和预测模型
- ⚡ **高性能**：支持大数据处理和分布式计算
- 🎯 **可视化**：直观的图表展示和交互式仪表盘
- 🔒 **可靠性**：完善的错误处理和数据质量控制

## 🎯 项目亮点

- **多源数据集成**：对接国家统计局、央行、海关等官方数据源，自动同步最新数据
- **智能预测引擎**：集成多种预测模型，自动选择最优方法，支持不确定性量化
- **大数据处理**：基于 Dask 和 Ray 实现分布式计算，支持处理大规模时间序列数据
- **交互式可视化**：支持多种图表类型，可导出为多种格式
- **经济模型库**：内置经典经济模型，如增长核算、奥肯定律、菲利普斯曲线等
- **可扩展性**：模块化设计，易于添加新数据源和分析方法

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
├── web_app.py
├── web_app_enhanced.py
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── analyzer.py         # 统计分析器
│   ├── big_data_processor.py  # 大数据处理器
│   ├── cache.py            # 缓存管理
│   ├── data_manager.py     # 数据管理器
│   ├── database.py         # 数据库管理
│   ├── data_sources/       # 数据源管理
│   │   ├── __init__.py
│   │   ├── base.py         # 数据源基类
│   │   ├── mock.py         # 模拟数据源
│   │   └── nbs.py          # 国家统计局数据源
│   ├── fitter.py           # 数据拟合器
│   ├── models/             # 经济模型
│   │   ├── __init__.py
│   │   ├── growth_accounting.py  # 增长核算模型
│   │   ├── okun_law.py     # 奥肯定律模型
│   │   └── phillips_curve.py  # 菲利普斯曲线模型
│   ├── predictor.py        # 高级预测器
│   ├── reporter.py         # 报告生成器
│   └── visualizer.py       # 可视化工具
├── models/                 # 数据模型
│   └── time_series.py      # 时间序列模型
└── data/                   # 数据存储
    ├── cache/              # 缓存文件
    ├── database/           # 数据库文件
    └── charts/             # 图表输出
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

### 1. 快速启动示例

```bash
# 运行演示模式
python start.py demo

# 预期输出：
# 🔧 正在初始化环境...
# ✅ 核心依赖检查通过
# 🧪 演示基本功能...
# 1. 获取GDP数据...
#    ✅ 成功获取 75 条GDP数据
#    时间范围: 2020-01-01 至 2026-03-01
#    统计信息: 均值=124.77, 标准差=13.98
# 2. 执行数据拟合...
#    ✅ 拟合完成，最佳方法: polynomial_4
#    最佳R² = 0.9425
# 3. 执行简单预测...
#    ✅ 简单预测下一个值: 147.76
```

### 2. 命令行界面
```bash
# 基础使用示例
python cli.py --help  # 查看命令帮助
python cli.py analyze --indicator gdp --time-range 2010-2024  # 分析GDP指标
python cli.py predict --indicator cpi --method ARIMA --forecast-period 12  # 预测CPI未来12期数据

# 增强版CLI使用
python module4_cli_enhancer.py --batch-process ./indicators_list.txt  # 批量处理指标列表
```

### 3. Python API使用

```python
from core.data_manager import DataManager
from core.analyzer import StatisticalAnalyzer

# 初始化数据管理器
dm = DataManager(use_mock_data=True)

# 获取GDP数据
gdp_ts = dm.fetch("gdp")
print(f"GDP数据: {len(gdp_ts)} 条记录")
print(f"时间范围: {gdp_ts.meta.start_date} 至 {gdp_ts.meta.end_date}")

# 初始化分析器
analyzer = StatisticalAnalyzer()

# 计算描述性统计
stats = analyzer.descriptive_stats(gdp_ts)
print("\nGDP描述性统计:")
print(stats)

# 计算增长率
growth_df = analyzer.calculate_growth_rates(gdp_ts)
print("\nGDP增长率:")
print(growth_df.tail())

# 关闭数据管理器
dm.close()
```

### 4. Web应用启动
```bash
# 启动增强版Web应用
python web_app_enhanced.py --host 0.0.0.0 --port 8080

# 访问地址: http://localhost:8080
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

