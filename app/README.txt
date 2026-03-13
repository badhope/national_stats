# 宏观经济数据分析平台 - 功能程序

本目录包含宏观经济数据分析平台的功能程序，可独立运行。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
# 完整分析流程
python main.py

# 仅加载数据
python main.py --data-only

# 仅执行分析
python main.py --analyze-only

# 查看版本
python main.py --version
```

## 主要功能

- 数据加载与预处理
- GDP/CPI/就业数据分析
- 可视化图表生成
- 趋势预测
- 报告生成

## 输出目录

程序运行后会在项目根目录生成：
- `output/` - 分析结果和图表
- `logs/` - 运行日志

## 更多信息

请参阅项目根目录的 README.md
