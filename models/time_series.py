"""
时间序列数据模型
封装 Pandas DataFrame，附加元数据和常用统计方法
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

# 导入配置中的指标定义
import sys
sys.path.append('..')
from config import IndicatorDefinition


@dataclass
class TimeSeriesMeta:
    """时间序列元数据"""
    indicator: IndicatorDefinition     # 指标定义
    source: str                        # 数据来源
    last_updated: datetime             # 最后更新时间
    start_date: Optional[str] = None   # 数据起始日期
    end_date: Optional[str] = None     # 数据结束日期
    count: int = 0                     # 数据点数量


class MacroTimeSeries:
    """宏观经济时间序列类"""
    
    def __init__(self, data: pd.DataFrame, meta: TimeSeriesMeta):
        """
        初始化时间序列
        
        Args:
            data: 包含 'date' 和 'value' 列的 DataFrame
            meta: 元数据对象
        """
        self._data = data.copy()
        self._meta = meta
        
        # 确保数据格式正确
        self._prepare_data()
    
    def _prepare_data(self):
        """准备数据：设置索引、排序、类型转换"""
        if 'date' in self._data.columns:
            self._data['date'] = pd.to_datetime(self._data['date'])
            self._data = self._data.set_index('date').sort_index()
        
        if 'value' in self._data.columns:
            self._data['value'] = pd.to_numeric(self._data['value'], errors='coerce')
        
        # 更新元数据
        if len(self._data) > 0:
            self._meta.start_date = str(self._data.index.min().date())
            self._meta.end_date = str(self._data.index.max().date())
            self._meta.count = len(self._data)
    
    @property
    def data(self) -> pd.DataFrame:
        """获取原始数据"""
        return self._data
    
    @property
    def meta(self) -> TimeSeriesMeta:
        """获取元数据"""
        return self._meta
    
    @property
    def values(self) -> pd.Series:
        """获取值序列"""
        return self._data['value']
    
    @property
    def name(self) -> str:
        """获取指标名称"""
        return self._meta.indicator.name
    
    # ==================== 基础统计方法 ====================
    
    def describe(self) -> pd.Series:
        """描述性统计"""
        return self.values.describe()
    
    def mean(self) -> float:
        """均值"""
        return self.values.mean()
    
    def std(self) -> float:
        """标准差"""
        return self.values.std()
    
    def min(self) -> float:
        """最小值"""
        return self.values.min()
    
    def max(self) -> float:
        """最大值"""
        return self.values.max()
    
    # ==================== 时间序列特有的统计方法 ====================
    
    def yoy(self, periods: int = 12) -> pd.Series:
        """
        同比增长率
        
        Args:
            periods: 周期数，月度数据默认12，季度数据默认4
        """
        # 根据频率自动调整周期
        freq = self._meta.indicator.frequency
        if freq == 'monthly':
            periods = 12
        elif freq == 'quarterly':
            periods = 4
        elif freq == 'yearly':
            return pd.Series(dtype=float)  # 年度数据无同比
        
        return self.values.pct_change(periods=periods) * 100
    
    def mom(self, periods: int = 1) -> pd.Series:
        """
        环比增长率
        
        Args:
            periods: 周期数，默认1
        """
        return self.values.pct_change(periods=periods) * 100
    
    def moving_average(self, window: int = 3) -> pd.Series:
        """
        移动平均
        
        Args:
            window: 窗口大小
        """
        return self.values.rolling(window=window).mean()
    
    def cumsum(self) -> pd.Series:
        """累计和"""
        return self.values.cumsum()
    
    def cumprod(self) -> pd.Series:
        """累计乘积"""
        return self.values.cumprod()
    
    # ==================== 数据处理方法 ====================
    
    def fillna(self, method: str = 'ffill') -> 'MacroTimeSeries':
        """
        填充缺失值
        
        Args:
            method: 填充方法，'ffill'(前向填充), 'bfill'(后向填充), 'mean'(均值填充)
        """
        filled_data = self._data.copy()
        
        if method == 'ffill':
            filled_data['value'] = filled_data['value'].fillna(method='ffill')
        elif method == 'bfill':
            filled_data['value'] = filled_data['value'].fillna(method='bfill')
        elif method == 'mean':
            filled_data['value'] = filled_data['value'].fillna(self.mean())
        
        return MacroTimeSeries(filled_data.reset_index(), self._meta)
    
    def resample(self, freq: str = 'Y', agg_func: str = 'sum') -> 'MacroTimeSeries':
        """
        重采样
        
        Args:
            freq: 目标频率，'Y'(年), 'Q'(季), 'M'(月)
            agg_func: 聚合函数，'sum', 'mean', 'last'
        """
        resampled = self._data.copy()
        
        if agg_func == 'sum':
            resampled = resampled.resample(freq).sum()
        elif agg_func == 'mean':
            resampled = resampled.resample(freq).mean()
        elif agg_func == 'last':
            resampled = resampled.resample(freq).last()
        
        # 更新元数据中的频率
        new_meta = TimeSeriesMeta(
            indicator=self._meta.indicator,
            source=self._meta.source,
            last_updated=datetime.now(),
            start_date=str(resampled.index.min().date()) if len(resampled) > 0 else None,
            end_date=str(resampled.index.max().date()) if len(resampled) > 0 else None,
            count=len(resampled)
        )
        
        return MacroTimeSeries(resampled.reset_index(), new_meta)
    
    # ==================== 数据筛选方法 ====================
    
    def filter_date(self, start: str, end: str) -> 'MacroTimeSeries':
        """
        按日期范围筛选
        
        Args:
            start: 开始日期，格式 '2020-01-01' 或 '2020'
            end: 结束日期
        """
        mask = (self._data.index >= start) & (self._data.index <= end)
        filtered_data = self._data.loc[mask].copy()
        
        return MacroTimeSeries(filtered_data.reset_index(), self._meta)
    
    # ==================== 导出方法 ====================
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """转换为字典列表"""
        result = []
        for idx, row in self._data.iterrows():
            result.append({
                'date': str(idx.date()) if hasattr(idx, 'date') else str(idx),
                'value': row['value']
            })
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return self._data.copy()
    
    def __repr__(self) -> str:
        return f"<MacroTimeSeries: {self.name} ({self._meta.start_date} to {self._meta.end_date}, {self._meta.count} points)>"
    
    def __len__(self) -> int:
        return len(self._data)


# ==================== 辅助函数 ====================

def create_time_series(
    indicator_code: str,
    data: pd.DataFrame,
    source: str = "unknown"
) -> MacroTimeSeries:
    """
    创建时间序列的便捷函数
    
    Args:
        indicator_code: 指标代码
        data: 数据 DataFrame
        source: 数据来源
    
    Returns:
        MacroTimeSeries 对象
    """
    from config import IndicatorLibrary
    
    indicator = IndicatorLibrary.get_indicator(indicator_code)
    
    meta = TimeSeriesMeta(
        indicator=indicator,
        source=source,
        last_updated=datetime.now()
    )
    
    return MacroTimeSeries(data, meta)
