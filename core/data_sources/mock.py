"""
模拟数据源
用于演示和测试，生成符合宏观经济规律的模拟数据
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import random

from .base import BaseDataSource
from models.time_series import MacroTimeSeries, TimeSeriesMeta, create_time_series
from config import IndicatorDefinition, IndicatorCategory, IndicatorFrequency


class MockDataSource(BaseDataSource):
    """模拟数据源 - 生成符合经济规律的测试数据"""
    
    def __init__(self):
        super().__init__()
        self.base_date = datetime(2020, 1, 1)
        # 设置随机种子以保证结果可重现
        np.random.seed(42)
        random.seed(42)
    
    def fetch(self, indicator_code: str, start_date: Optional[str] = None,
              end_date: Optional[str] = None, **kwargs) -> MacroTimeSeries:
        """
        生成模拟数据
        
        Args:
            indicator_code: 指标代码
            start_date: 开始日期
            end_date: 结束日期
        """
        # 确定数据范围
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m")
        else:
            start_dt = self.base_date
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m")
        else:
            end_dt = datetime.now()
        
        # 计算月份数
        months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        num_points = max(12, months_diff + 1)  # 至少12个数据点
        
        # 生成日期序列
        dates = [start_dt + timedelta(days=30*i) for i in range(num_points)]
        dates = [d.replace(day=1) for d in dates]  # 统一到每月1日
        
        # 根据指标类型生成相应的数据模式
        values = self._generate_indicator_data(indicator_code, num_points)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # 创建时间序列对象
        meta = self._get_indicator_meta(indicator_code)
        ts = MacroTimeSeries(df, meta)
        
        return ts
    
    def _generate_indicator_data(self, indicator_code: str, num_points: int) -> list:
        """根据指标类型生成数据"""
        
        if indicator_code == "gdp":
            # GDP数据：长期增长趋势 + 周期性波动 + 随机噪声
            trend = np.linspace(100, 150, num_points)  # 基础趋势
            cycle = 5 * np.sin(np.linspace(0, 4*np.pi, num_points))  # 周期波动
            noise = np.random.normal(0, 2, num_points)  # 随机噪声
            values = trend + cycle + noise
            return np.maximum(values, 80).tolist()  # 确保正值
            
        elif indicator_code == "cpi":
            # CPI数据：温和上涨 + 季节性 + 波动
            base = 100
            trend = np.linspace(0, 15, num_points)  # 温和通胀
            seasonal = 2 * np.sin(2 * np.pi * np.arange(num_points) / 12)  # 季节性
            noise = np.random.normal(0, 0.5, num_points)
            values = base + trend + seasonal + noise
            return np.maximum(values, 95).tolist()
            
        elif indicator_code == "ppi":
            # PPI数据：波动较大
            base = 100
            trend = np.linspace(0, 10, num_points)
            volatility = 8 * np.random.randn(num_points)
            values = base + trend + volatility
            return np.maximum(values, 80).tolist()
            
        elif indicator_code == "pmi_manufacturing":
            # PMI数据：围绕50波动
            base = 50
            trend = np.linspace(0, 3, num_points)
            cyclical = 8 * np.sin(np.linspace(0, 3*np.pi, num_points))
            noise = np.random.normal(0, 2, num_points)
            values = base + trend + cyclical + noise
            # 确保在合理范围内
            return np.clip(values, 40, 60).tolist()
            
        elif indicator_code == "retail_sales_yoy":
            # 零售销售增长率：正负波动
            trend = np.linspace(2, 8, num_points)
            cyclical = 6 * np.sin(np.linspace(0, 4*np.pi, num_points))
            noise = np.random.normal(0, 2, num_points)
            values = trend + cyclical + noise
            return values.tolist()
            
        elif indicator_code == "fixed_asset_investment_yoy":
            # 固定资产投资增长率
            trend = np.linspace(5, 12, num_points)
            volatility = np.random.normal(0, 3, num_points)
            values = trend + volatility
            return np.maximum(values, -5).tolist()  # 不会让增长率过于负面
            
        elif indicator_code == "export":
            # 出口数据：增长趋势 + 波动
            base = 2000  # 亿美元
            trend = np.linspace(0, 800, num_points)
            cyclical = 200 * np.sin(np.linspace(0, 3*np.pi, num_points))
            noise = np.random.normal(0, 50, num_points)
            values = base + trend + cyclical + noise
            return np.maximum(values, 1500).tolist()
            
        elif indicator_code == "m2":
            # M2货币供应量：稳定增长
            base = 200  # 万亿元
            trend = np.linspace(0, 100, num_points)
            seasonal = 5 * np.sin(2 * np.pi * np.arange(num_points) / 12)
            noise = np.random.normal(0, 2, num_points)
            values = base + trend + seasonal + noise
            return np.maximum(values, 150).tolist()
            
        else:
            # 默认：简单增长 + 噪声
            base = 100
            trend = np.linspace(0, 20, num_points)
            noise = np.random.normal(0, 3, num_points)
            values = base + trend + noise
            return np.maximum(values, 50).tolist()
    
    def _get_indicator_meta(self, indicator_code: str) -> TimeSeriesMeta:
        """获取指标元数据"""
        meta_map = {
            "gdp": ("国内生产总值", IndicatorCategory.PRODUCTION, IndicatorFrequency.QUARTERLY, "亿元", "mock"),
            "cpi": ("居民消费价格指数", IndicatorCategory.PRICE, IndicatorFrequency.MONTHLY, "上年同月=100", "mock"),
            "ppi": ("工业生产者出厂价格指数", IndicatorCategory.PRICE, IndicatorFrequency.MONTHLY, "上年同月=100", "mock"),
            "pmi_manufacturing": ("制造业PMI", IndicatorCategory.PRODUCTION, IndicatorFrequency.MONTHLY, "%", "mock"),
            "retail_sales_yoy": ("社会消费品零售总额同比增长率", IndicatorCategory.DEMAND, IndicatorFrequency.MONTHLY, "%", "mock"),
            "fixed_asset_investment_yoy": ("固定资产投资同比增长率", IndicatorCategory.DEMAND, IndicatorFrequency.MONTHLY, "%", "mock"),
            "export": ("出口总额", IndicatorCategory.TRADE, IndicatorFrequency.MONTHLY, "亿美元", "mock"),
            "m2": ("货币供应量M2", IndicatorCategory.MONEY_SUPPLY, IndicatorFrequency.MONTHLY, "万亿元", "mock")
        }
        
        name, category, frequency, unit, source = meta_map.get(indicator_code, 
                                                              (indicator_code, IndicatorCategory.PRODUCTION, 
                                                               IndicatorFrequency.MONTHLY, "单位", "mock"))
        
        # 创建指标定义
        indicator_def = IndicatorDefinition(
            code=indicator_code,
            name=name,
            category=category,
            frequency=frequency,
            unit=unit,
            source=source
        )
        
        return TimeSeriesMeta(
            indicator=indicator_def,
            source=source,
            last_updated=datetime.now()
        )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建模拟数据源
    mock_source = MockDataSource()
    
    # 生成不同类型的数据
    indicators = ["gdp", "cpi", "pmi_manufacturing", "retail_sales_yoy"]
    
    for indicator in indicators:
        print(f"\n=== {indicator.upper()} 模拟数据 ===")
        ts = mock_source.fetch(indicator, "2020-01", "2023-12")
        print(f"数据点数: {len(ts)}")
        print(f"时间范围: {ts.meta.start_date} 至 {ts.meta.end_date}")
        print(f"统计摘要:")
        print(f"  均值: {ts.data['value'].mean():.2f}")
        print(f"  标准差: {ts.data['value'].std():.2f}")
        print(f"  最小值: {ts.data['value'].min():.2f}")
        print(f"  最大值: {ts.data['value'].max():.2f}")