#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列数据模型
宏观经济数据分析平台 - 数据结构定义
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TimeSeriesMeta:
    """时间序列元数据"""
    indicator: str
    source: str
    unit: str
    frequency: str = "年度"
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator": self.indicator,
            "source": self.source,
            "unit": self.unit,
            "frequency": self.frequency,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class MacroTimeSeries:
    """宏观经济时间序列数据"""
    meta: TimeSeriesMeta
    dates: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add(self, date: int, value: float):
        """添加数据点"""
        self.dates.append(date)
        self.values.append(value)
    
    def get_array(self) -> np.ndarray:
        """获取numpy数组"""
        return np.array(self.values)
    
    def get_date_array(self) -> np.ndarray:
        """获取日期数组"""
        return np.array(self.dates)
    
    def latest(self) -> Optional[float]:
        """获取最新值"""
        return self.values[-1] if self.values else None
    
    def earliest(self) -> Optional[float]:
        """获取最早值"""
        return self.values[0] if self.values else None
    
    def growth_rate(self) -> List[float]:
        """计算增长率"""
        if len(self.values) < 2:
            return []
        rates = []
        for i in range(1, len(self.values)):
            if self.values[i-1] != 0:
                rate = (self.values[i] - self.values[i-1]) / self.values[i-1] * 100
                rates.append(rate)
            else:
                rates.append(0.0)
        return rates
    
    def mean(self) -> float:
        """计算平均值"""
        return float(np.mean(self.values)) if self.values else 0.0
    
    def std(self) -> float:
        """计算标准差"""
        return float(np.std(self.values)) if self.values else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "dates": self.dates,
            "values": self.values,
            "metadata": self.metadata
        }


def create_time_series(
    indicator: str,
    source: str,
    unit: str,
    dates: List[int],
    values: List[float],
    frequency: str = "年度"
) -> MacroTimeSeries:
    """创建时间序列数据的便捷函数"""
    meta = TimeSeriesMeta(
        indicator=indicator,
        source=source,
        unit=unit,
        frequency=frequency,
        last_updated=datetime.now()
    )
    
    ts = MacroTimeSeries(meta=meta)
    for date, value in zip(dates, values):
        ts.add(date, value)
    
    return ts
