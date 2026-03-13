# 宏观经济数据分析平台

专业的宏观经济数据分析和可视化平台。

## 功能特点

- 多维数据可视化
- 一键数据导出
- 智能趋势预测
- 数据安全保障

## 快速开始

### 查看产品介绍

直接在浏览器中打开 `docs-presentation/index.html`

或使用本地服务器：

```bash
cd docs-presentation
python -m http.server 8080
# 访问 http://localhost:8080
```

### 运行功能程序

```bash
# 安装依赖
pip install -r app/requirements.txt

# 运行程序
python app/main.py

# 查看帮助
python app/main.py --help
```

## 项目结构

```
national_stats/
├── app/                    # 功能程序
│   ├── main.py
│   └── requirements.txt
├── docs-presentation/       # 产品介绍网页
│   └── index.html
├── core/                   # 核心模块
└── ...
```

## 许可证

MIT License
