"""
全局配置文件
定义常量、数据源映射和系统设置
"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 数据存储路径
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPORT_DIR = DATA_DIR / "exports"
CHART_DIR = DATA_DIR / "charts"
LOG_DIR = BASE_DIR / "logs"

# 确保目录存在
for d in [DATA_DIR, CACHE_DIR, EXPORT_DIR, CHART_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 爬虫配置
SCRAPER_CONFIG = {
    'timeout': 30,
    'retry_times': 3,
    'delay': 0.5,  # 请求间隔(秒)
    'concurrent_limit': 5,  # 最大并发数
    'cache_expire_hours': 24  # 缓存过期时间
}

# 数据指标定义
METRICS = {
    'gdp': {
        'name': '国内生产总值',
        'unit': '亿元',
        'api_code': 'A01',
        'category': '国民经济'
    },
    'population': {
        'name': '人口数据',
        'unit': '万人',
        'api_code': 'A02',
        'category': '人口就业'
    },
    'cpi': {
        'name': '居民消费价格指数',
        'unit': '%',
        'api_code': 'A03',
        'category': '价格指数'
    },
    'trade': {
        'name': '进出口贸易',
        'unit': '亿美元',
        'api_code': 'A04',
        'category': '对外经济'
    },
    'industry': {
        'name': '工业增加值',
        'unit': '%',
        'api_code': 'A05',
        'category': '工业经济'
    },
    'investment': {
        'name': '固定资产投资',
        'unit': '亿元',
        'api_code': 'A06',
        'category': '固定资产投资'
    },
    'retail': {
        'name': '社会消费品零售总额',
        'unit': '亿元',
        'api_code': 'A07',
        'category': '国内贸易'
    },
    'income': {
        'name': '居民人均可支配收入',
        'unit': '元',
        'api_code': 'A08',
        'category': '人民生活'
    }
}

# 可视化配色方案
THEMES = {
    'professional': ['#2E86AB', '#A23B72', '#28A745', '#FFC107', '#DC3545', '#17A2B8'],
    'pastel': ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#B5EAD7', '#FFDAC1'],
    'dark': ['#1B1B1E', '#373740', '#5C5C66', '#898994', '#B4B4BE', '#FFFFFF']
}
