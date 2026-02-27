# 📊 宏观经济智能分析平台 (National Statistics Intelligence Platform)

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
├── core/                    # 核心模块
│   ├── data_sources/       # 数据源管理
│   │   ├── base.py         # 数据源基类
│   │   └── nbs.py          # 国家统计局数据源
│   ├── models/             # 经济模型
│   │   ├── growth_accounting.py  # 增长核算模型
│   │   ├── okun_law.py     # 奥肯定律
│   │   └── phillips_curve.py     # 菲利普斯曲线
│   ├── analyzer.py         # 统计分析器
│   ├── cache.py            # 缓存管理
│   ├── data_manager.py     # 数据管理器
│   ├── database.py         # 数据库管理
│   ├── predictor.py        # 高级预测器
│   ├── fitter.py           # 数据拟合器
│   ├── big_data_processor.py  # 大数据处理器
│   ├── visualizer.py       # 可视化工具
│   └── reporter.py         # 报告生成器
├── models/                 # 数据模型
│   └── time_series.py      # 时间序列模型
├── config.py               # 全局配置
├── cli.py                  # 命令行接口
├── web_app.py              # 基础Web应用
├── web_app_enhanced.py     # 增强版Web应用
└── requirements.txt        # 依赖包列表
```

## 🛠️ 安装与配置

### 环境要求
- Python 3.9+
- 8GB+ 内存推荐
- 现代CPU（支持多核处理）

### 快速安装

```bash
# 克隆项目
git clone <repository-url>
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
# 获取数据
python cli.py fetch gdp --start 2020-01 --end 2023-12

# 数据分析
python cli.py analyze cpi --trend --report

# 智能预测
python cli.py predict gdp --periods 12 --method auto

# 数据拟合
python cli.py fit gdp --methods polynomial,exponential --extrapolate 6

# 多指标对比
python cli.py compare gdp cpi pmi_manufacturing --pca --report

# 经济模型分析
python cli.py model growth_accounting --capital-share 0.4

# 查看可用指标
python cli.py list-indicators --category production
```

### 2. Web应用

#### 基础版本
```bash
streamlit run web_app.py
```

#### 增强版本（推荐）
```bash
streamlit run web_app_enhanced.py
```

访问 `http://localhost:8501` 使用图形界面。

### 3. 编程接口

```python
from core import DataManager, Predictor, AdvancedFitter
from core.big_data_processor import BigDataProcessor

# 数据获取
dm = DataManager()
gdp_data = dm.fetch('gdp')

# 智能预测
predictor = Predictor(method='auto')
forecast_result = predictor.forecast(gdp_data, periods=12)

# 数据拟合
fitter = AdvancedFitter()
fit_result = fitter.fit(x_data, y_data)

# 大数据分析
processor = BigDataProcessor()
batch_data = processor.batch_process_indicators(['gdp', 'cpi', 'ppi'])
analysis_result = processor.process_large_dataset(
    batch_data['successful_data'], 
    operations=['correlation', 'clustering']
)
```

## 📊 支持的指标体系

### 生产类指标
- 国内生产总值(GDP)及其增长率
- 工业增加值增长率
- 制造业PMI

### 价格类指标
- 居民消费价格指数(CPI)
- 工业生产者价格指数(PPI)
- 商品零售价格指数

### 需求类指标
- 固定资产投资增长率
- 社会消费品零售总额增长率
- 出口进口总额

### 货币金融类
- 货币供应量(M0/M1/M2)
- 银行间同业拆借利率
- 信贷投放数据

### 就业类指标
- 城镇调查失业率
- 新增就业人数
- 求职人数

## 🔧 技术架构

### 数据层
- **数据源适配器**: 统一接口对接不同官方数据源
- **智能缓存**: Redis + 本地文件系统双重缓存
- **数据库抽象**: SQLAlchemy ORM支持多种数据库

### 计算层
- **并行处理**: ThreadPoolExecutor + ProcessPoolExecutor
- **分布式计算**: Dask数据框 + Ray任务调度
- **内存优化**: 分块处理 + 惰性计算

### 算法层
- **传统统计**: Statsmodels时间序列分析
- **机器学习**: Scikit-learn + XGBoost + LightGBM
- **深度学习**: PyTorch/TensorFlow神经网络
- **专业模型**: Prophet + ARIMA + 自研经济模型

### 应用层
- **CLI工具**: argparse驱动的命令行界面
- **Web界面**: Streamlit构建的交互式仪表盘
- **API服务**: FastAPI支持的RESTful接口

## 📈 性能特点

### 处理能力
- **单机性能**: 支持万级别时间序列同时处理
- **分布式扩展**: 可扩展至集群级别的数据处理
- **实时响应**: 关键查询毫秒级响应

### 内存效率
- **流式处理**: 大文件分块读取
- **压缩存储**: 数据自动压缩存储
- **智能缓存**: LRU策略优化内存使用

## 🔒 安全与合规

- **数据源认证**: 官方API密钥管理
- **访问控制**: 用户权限分级管理
- **审计日志**: 完整操作记录
- **隐私保护**: 敏感数据脱敏处理

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码质量检查
flake8 core/
black core/
```

### 贡献流程
1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- **项目维护者**: [Your Name]
- **邮箱**: [your.email@example.com]
- **项目主页**: [https://github.com/username/national_stats]

## 🙏 致谢

感谢以下开源项目的贡献：
- [Statsmodels](https://www.statsmodels.org/) - 统计建模
- [Scikit-learn](https://scikit-learn.org/) - 机器学习
- [Streamlit](https://streamlit.io/) - Web应用框架
- [Dask](https://dask.org/) - 并行计算
- [Ray](https://www.ray.io/) - 分布式系统

---
*Made with ❤️ for economic research and policy analysis*