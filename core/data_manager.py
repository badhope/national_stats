"""
数据管理器
整合数据源、缓存和数据库，提供统一的数据访问接口
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# 导入各模块
import sys
sys.path.append('..')
from config import Config, IndicatorLibrary, IndicatorDefinition
from models.time_series import MacroTimeSeries, TimeSeriesMeta
from .cache import CacheManager
from .database import DatabaseManager
from .data_sources.nbs import NBSDataSource
from .data_sources.mock import MockDataSource


class DataManager:
    """数据管理器 - 核心数据访问层"""
    
    def __init__(self, use_cache: bool = True, use_database: bool = True, 
                 use_mock_data: bool = True):
        """
        初始化数据管理器
        
        Args:
            use_cache: 是否使用缓存
            use_database: 是否使用数据库
            use_mock_data: 是否使用模拟数据（用于演示）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化各组件
        self.cache = CacheManager() if use_cache else None
        self.database = DatabaseManager() if use_database else None
        
        # 初始化数据源
        self.data_sources = {
            'nbs': NBSDataSource(),
            'mock': MockDataSource()  # 添加模拟数据源
        }
        
        self.use_mock_data = use_mock_data
        self.logger.info("数据管理器初始化完成")
    
    def fetch(self, indicator_code: str, 
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              force_refresh: bool = False) -> Optional[MacroTimeSeries]:
        """
        获取时间序列数据（智能优先级：数据库 -> 缓存 -> 数据源）
        
        Args:
            indicator_code: 指标代码
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 是否强制刷新（跳过缓存和数据库）
        
        Returns:
            时间序列对象
        """
        self.logger.info(f"获取数据: {indicator_code}")
        
        # 1. 如果强制刷新，直接从数据源获取
        if force_refresh:
            return self._fetch_from_source(indicator_code, start_date, end_date)
        
        # 2. 尝试从数据库获取
        if self.database:
            ts = self.database.load_time_series(indicator_code, start_date, end_date)
            if ts is not None:
                self.logger.info(f"从数据库获取: {indicator_code}")
                return ts
        
        # 3. 尝试从缓存获取
        if self.cache:
            cache_key = f"ts_{indicator_code}_{start_date}_{end_date}"
            ts = self.cache.get(cache_key)
            if ts is not None:
                self.logger.info(f"从缓存获取: {indicator_code}")
                return ts
        
        # 4. 从数据源获取
        return self._fetch_from_source(indicator_code, start_date, end_date)
    
    def _fetch_from_source(self, indicator_code: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[MacroTimeSeries]:
        """从数据源获取数据"""
        try:
            # 获取指标定义
            try:
                indicator = IndicatorLibrary.get_indicator(indicator_code)
                source_key = indicator.source
            except ValueError:
                # 如果指标不存在，使用默认设置
                source_key = 'mock' if self.use_mock_data else 'nbs'
            
            # 选择数据源
            if self.use_mock_data and source_key in ['nbs', 'mock']:
                # 优先使用模拟数据进行演示
                data_source = self.data_sources['mock']
                source_key = 'mock'
            elif source_key in self.data_sources:
                data_source = self.data_sources[source_key]
            else:
                self.logger.error(f"不支持的数据源: {source_key}")
                # 回退到模拟数据源
                data_source = self.data_sources['mock']
                source_key = 'mock'
            
            # 获取数据
            ts = data_source.fetch(indicator_code, start_date, end_date)
            
            # 保存到数据库
            if self.database and source_key != 'mock':
                self.database.save_time_series(ts)
            
            # 保存到缓存
            if self.cache:
                cache_key = f"ts_{indicator_code}_{start_date}_{end_date}"
                self.cache.set(cache_key, ts)
            
            self.logger.info(f"从{source_key}数据源获取并保存: {indicator_code}")
            return ts
        
        except Exception as e:
            self.logger.error(f"获取数据失败: {e}")
            return None
    
    def fetch_multiple(self, indicator_codes: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, MacroTimeSeries]:
        """
        批量获取多个指标的数据
        
        Args:
            indicator_codes: 指标代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            指标代码到时间序列的映射字典
        """
        result = {}
        
        for code in indicator_codes:
            ts = self.fetch(code, start_date, end_date)
            if ts is not None:
                result[code] = ts
        
        return result
    
    def get_indicator_info(self, indicator_code: str) -> Dict[str, Any]:
        """
        获取指标信息
        
        Args:
            indicator_code: 指标代码
        
        Returns:
            指标信息字典
        """
        try:
            indicator = IndicatorLibrary.get_indicator(indicator_code)
            
            info = {
                'code': indicator.code,
                'name': indicator.name,
                'category': indicator.category,
                'frequency': indicator.frequency,
                'unit': indicator.unit,
                'source': indicator.source,
                'description': indicator.description,
                'is_leading': indicator.is_leading,
                'is_coincident': indicator.is_coincident,
                'is_lagging': indicator.is_lagging
            }
            
            # 如果数据库中有数据，添加统计信息
            if self.database:
                ts = self.database.load_time_series(indicator_code)
                if ts:
                    info['data_count'] = len(ts)
                    info['start_date'] = ts.meta.start_date
                    info['end_date'] = ts.meta.end_date
            
            return info
        
        except ValueError:
            return {}
    
    def list_indicators(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有指标
        
        Args:
            category: 筛选类别，如果为 None 则返回所有
        
        Returns:
            指标信息列表
        """
        indicators = IndicatorLibrary.NBS_INDICATORS
        
        if category:
            indicators = [ind for ind in indicators if ind.category == category]
        
        return [
            {
                'code': ind.code,
                'name': ind.name,
                'category': ind.category,
                'frequency': ind.frequency,
                'unit': ind.unit
            }
            for ind in indicators
        ]
    
    def refresh_data(self, indicator_codes: Optional[List[str]] = None):
        """
        刷新数据（强制从数据源重新获取）
        
        Args:
            indicator_codes: 要刷新的指标列表，如果为 None 则刷新所有
        """
        if indicator_codes is None:
            # 获取所有支持的指标
            indicator_codes = [ind.code for ind in IndicatorLibrary.NBS_INDICATORS]
        
        for code in indicator_codes:
            self.logger.info(f"正在刷新: {code}")
            self.fetch(code, force_refresh=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            'cache': self.cache.get_cache_stats() if self.cache else None,
            'database': self.database.get_database_stats() if self.database else None
        }
        
        return stats
    
    def close(self):
        """关闭所有连接"""
        for source in self.data_sources.values():
            source.close()
        
        self.logger.info("数据管理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建数据管理器
    with DataManager(use_mock_data=True) as dm:
        # 获取单个指标
        print("获取 GDP 数据...")
        gdp_ts = dm.fetch("gdp")
        if gdp_ts:
            print(gdp_ts)
            print(gdp_ts.describe())
        
        # 批量获取多个指标
        print("\n批量获取多个指标...")
        indicators = ["gdp", "cpi", "pmi_manufacturing"]
        data = dm.fetch_multiple(indicators)
        
        for code, ts in data.items():
            print(f"{code}: {len(ts)} 条数据")
        
        # 列出所有指标
        print("\n所有生产类指标:")
        production_indicators = dm.list_indicators(category="production")
        for ind in production_indicators:
            print(f"  {ind['code']}: {ind['name']}")