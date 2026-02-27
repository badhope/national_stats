"""
全局配置文件
定义项目的所有配置参数，包括路径、数据源、爬虫、数据库、可视化等
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import yaml


# ==================== 路径配置 ====================
class PathConfig:
    """路径配置类"""
    # 项目根目录
    BASE_DIR = Path(__file__).parent
    
    # 数据目录
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = DATA_DIR / "cache"
    EXPORT_DIR = DATA_DIR / "exports"
    DATABASE_DIR = DATA_DIR / "database"
    CHART_DIR = DATA_DIR / "charts"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CONFIG_DIR = BASE_DIR / "config"
    
    # 数据库文件
    DATABASE_FILE = DATABASE_DIR / "national_stats.db"
    REDIS_CACHE_FILE = CACHE_DIR / "redis_cache.rdb"
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有必要目录存在"""
        dirs = [
            cls.DATA_DIR,
            cls.CACHE_DIR,
            cls.EXPORT_DIR,
            cls.DATABASE_DIR,
            cls.CHART_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.CONFIG_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ==================== 爬虫配置 ====================
@dataclass
class ScraperConfig:
    """爬虫配置"""
    timeout: int = 30
    retry_times: int = 3
    retry_delay: float = 1.0
    request_delay: float = 0.5
    max_workers: int = 10
    cache_expire_hours: int = 24
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    proxies: Dict[str, str] = field(default_factory=dict)
    rate_limit: int = 10  # 每秒请求数限制
    
    def __post_init__(self):
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy:
            self.proxies["http"] = http_proxy
        if https_proxy:
            self.proxies["https"] = https_proxy


# ==================== 数据库配置 ====================
@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_type: str = "sqlite"
    connection_string: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False
    
    def __post_init__(self):
        if not self.connection_string:
            if self.db_type == "sqlite":
                self.connection_string = f"sqlite:///{PathConfig.DATABASE_FILE}"
            elif self.db_type == "postgresql":
                self.connection_string = "postgresql://user:password@localhost/national_stats"
            elif self.db_type == "mysql":
                self.connection_string = "mysql+pymysql://user:password@localhost/national_stats"


# ==================== Redis缓存配置 ====================
@dataclass
class RedisConfig:
    """Redis缓存配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    expire_seconds: int = 86400  # 24小时
    enabled: bool = True


# ==================== 指标配置 ====================
class IndicatorCategory:
    PRODUCTION = "production"
    DEMAND = "demand"
    PRICE = "price"
    EMPLOYMENT = "employment"
    FINANCE = "finance"
    TRADE = "trade"
    FISCAL = "fiscal"
    REAL_ESTATE = "real_estate"
    INCOME = "income"
    MONEY_SUPPLY = "money_supply"
    INDUSTRY = "industry"
    CONSUMER = "consumer"


class IndicatorFrequency:
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class IndicatorDefinition:
    """指标定义"""
    code: str
    name: str
    category: str
    frequency: str
    unit: str
    source: str
    description: str = ""
    is_leading: bool = False
    is_coincident: bool = False
    is_lagging: bool = False
    parent_code: str = ""
    weight: float = 1.0
    api_endpoint: str = ""
    data_format: str = "json"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code, "name": self.name, "category": self.category,
            "frequency": self.frequency, "unit": self.unit, "source": self.source,
            "description": self.description, "is_leading": self.is_leading,
            "is_coincident": self.is_coincident, "is_lagging": self.is_lagging,
            "parent_code": self.parent_code, "weight": self.weight,
            "api_endpoint": self.api_endpoint, "data_format": self.data_format
        }


# 扩展指标库
class IndicatorLibrary:
    NBS_INDICATORS: List[IndicatorDefinition] = [
        # GDP相关
        IndicatorDefinition("gdp", "国内生产总值", IndicatorCategory.PRODUCTION, 
                          IndicatorFrequency.QUARTERLY, "亿元", "nbs", 
                          is_coincident=True, api_endpoint="/api/gdp"),
        IndicatorDefinition("gdp_yoy", "GDP同比增长率", IndicatorCategory.PRODUCTION,
                          IndicatorFrequency.QUARTERLY, "%", "nbs", 
                          is_coincident=True, api_endpoint="/api/gdp/yoy"),
        
        # 工业生产
        IndicatorDefinition("industrial_yoy", "工业增加值同比增长率", IndicatorCategory.PRODUCTION,
                          IndicatorFrequency.MONTHLY, "%", "nbs",                          is_coincident=True, api_endpoint="/api/industry/value_added"),
        
        # 价格指数
        IndicatorDefinition("cpi", "居民消费价格指数", IndicatorCategory.PRICE,
                          IndicatorFrequency.MONTHLY, "上年同月=100", "nbs", 
                          is_coincident=True, api_endpoint="/api/price/cpi"),
        IndicatorDefinition("ppi", "工业生产者出厂价格指数", IndicatorCategory.PRICE,
                          IndicatorFrequency.MONTHLY, "上年同月=100", "nbs", 
                          is_leading=True, api_endpoint="/api/price/ppi"),
        
        # 投资消费
        IndicatorDefinition("fixed_asset_investment_yoy", "固定资产投资同比增长率", 
                          IndicatorCategory.DEMAND, IndicatorFrequency.MONTHLY, "%", "nbs", 
                          is_leading=True, api_endpoint="/api/investment/fixed_asset"),
        IndicatorDefinition("retail_sales_yoy", "社会消费品零售总额同比增长率",
                          IndicatorCategory.DEMAND, IndicatorFrequency.MONTHLY, "%", "nbs", 
                          is_coincident=True, api_endpoint="/api/consumption/retail_sales"),
        
        # 对外贸易
        IndicatorDefinition("export", "出口总额", IndicatorCategory.TRADE,
                          IndicatorFrequency.MONTHLY, "亿美元", "customs", 
                          is_coincident=True, api_endpoint="/api/trade/export"),
        IndicatorDefinition("import", "进口总额", IndicatorCategory.TRADE,
                          IndicatorFrequency.MONTHLY, "亿美元", "customs", 
                          is_coincident=True, api_endpoint="/api/trade/import"),
        
        # 制造业PMI
        IndicatorDefinition("pmi_manufacturing", "制造业PMI", IndicatorCategory.PRODUCTION,
                          IndicatorFrequency.MONTHLY, "%", "nbs", 
                          is_leading=True, weight=1.5, api_endpoint="/api/pmi/manufacturing"),
        
        # 货币供应量
        IndicatorDefinition("m0", "货币供应量M0", IndicatorCategory.MONEY_SUPPLY,
                          IndicatorFrequency.MONTHLY, "万亿元", "pbc", 
                          api_endpoint="/api/money_supply/m0"),
        IndicatorDefinition("m1", "货币供应量M1", IndicatorCategory.MONEY_SUPPLY,
                          IndicatorFrequency.MONTHLY, "万亿元", "pbc", 
                          api_endpoint="/api/money_supply/m1"),
        IndicatorDefinition("m2", "货币供应量M2", IndicatorCategory.MONEY_SUPPLY,
                          IndicatorFrequency.MONTHLY, "万亿元", "pbc", 
                          api_endpoint="/api/money_supply/m2"),
        
        # 就业
        IndicatorDefinition("urban_unemployment_rate", "城镇调查失业率", IndicatorCategory.EMPLOYMENT,
                          IndicatorFrequency.MONTHLY, "%", "nbs", 
                          is_lagging=True, api_endpoint="/api/employment/unemployment_rate"),
        
        # 房地产
        IndicatorDefinition("house_price_index", "70个大中城市房价指数", IndicatorCategory.REAL_ESTATE,
                          IndicatorFrequency.MONTHLY, "上年同月=100", "nbs", 
                          api_endpoint="/api/real_estate/price_index"),
    ]
    
    @classmethod
    def get_indicator(cls, code: str) -> IndicatorDefinition:
        for ind in cls.NBS_INDICATORS:
            if ind.code == code:
                return ind
        raise ValueError(f"未找到指标: {code}")
    
    @classmethod
    def get_leading_indicators(cls) -> List[IndicatorDefinition]:
        return [ind for ind in cls.NBS_INDICATORS if ind.is_leading]
    
    @classmethod
    def get_coincident_indicators(cls) -> List[IndicatorDefinition]:
        return [ind for ind in cls.NBS_INDICATORS if ind.is_coincident]
    
    @classmethod
    def get_lagging_indicators(cls) -> List[IndicatorDefinition]:
        return [ind for ind in cls.NBS_INDICATORS if ind.is_lagging]


# ==================== 模型配置 ====================
@dataclass
class ModelConfig:
    """机器学习模型配置"""
    # 时间序列预测
    forecast_horizon: int = 24  # 预测期数
    validation_split: float = 0.2  # 验证集比例
    test_split: float = 0.1  # 测试集比例
    
    # Prophet参数
    prophet_seasonality_mode: str = "multiplicative"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = False
    prophet_daily_seasonality: bool = False
    
    # ARIMA参数
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    
    # XGBoost参数
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # 神经网络参数
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    nn_epochs: int = 100
    nn_batch_size: int = 32
    nn_learning_rate: float = 0.001


# ==================== 可视化配置 ====================
@dataclass
class VisualizationConfig:
    figure_width: int = 12
    figure_height: int = 8
    chinese_font: str = "SimHei"
    color_palette: str = "husl"
    theme: str = "default"
    interactive: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg", "html"])
    dpi: int = 300


# ==================== 分析配置 ====================
@dataclass
class AnalysisConfig:
    ma_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    forecast_years: int = 5
    correlation_method: str = "pearson"
    significance_level: float = 0.05
    outlier_detection: bool = True
    seasonal_decomposition: bool = True


# ==================== 日志配置 ====================
@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_rotation: str = "10 MB"
    retention_days: int = 30
    enable_console: bool = True


# ==================== 性能配置 ====================
@dataclass
class PerformanceConfig:
    """性能优化配置"""
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    memory_limit_gb: float = 8.0
    cache_enabled: bool = True
    compression_enabled: bool = True


# ==================== 大数据配置 ====================
@dataclass
class BigDataConfig:
    """大数据处理配置"""
    use_dask: bool = True
    use_ray: bool = True
    batch_size: int = 50
    max_concurrent_requests: int = 20
    distributed_computing: bool = True
    data_partition_size: int = 10000
    enable_caching: bool = True


# ==================== 全局配置管理 ====================
class Config:
    paths = PathConfig
    scraper = ScraperConfig()
    database = DatabaseConfig()
    redis = RedisConfig()
    model = ModelConfig()
    visualization = VisualizationConfig()
    analysis = AnalysisConfig()
    log = LogConfig()
    performance = PerformanceConfig()
    big_data = BigDataConfig()
    
    @classmethod
    def initialize(cls):
        """初始化配置"""
        PathConfig.ensure_dirs()
        
        # 设置matplotlib中文显示
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = [cls.visualization.chinese_font]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 加载自定义配置文件（如果存在）
        config_file = PathConfig.CONFIG_DIR / "config.yaml"
        if config_file.exists():
            cls._load_custom_config(config_file)
    
    @classmethod
    def _load_custom_config(cls, config_file: Path):
        """加载自定义配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f)
            
            # 更新配置
            for section, values in custom_config.items():
                if hasattr(cls, section):
                    section_obj = getattr(cls, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
        except Exception as e:
            print(f"加载自定义配置失败: {e}")

# 初始化配置
Config.initialize()