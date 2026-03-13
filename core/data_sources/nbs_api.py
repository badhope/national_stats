#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家统计局数据源模块
宏观经济数据分析平台 - 数据获取

支持从国家统计局API获取真实数据
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceBase(ABC):
    """数据源基类"""
    
    @abstractmethod
    def fetch_gdp(self) -> Dict[str, Any]:
        """获取GDP数据"""
        pass
    
    @abstractmethod
    def fetch_cpi(self) -> Dict[str, Any]:
        """获取CPI数据"""
        pass
    
    @abstractmethod
    def fetch_employment(self) -> Dict[str, Any]:
        """获取就业数据"""
        pass


class NBSDataSource(DataSourceBase):
    """国家统计局数据源
    
    注意: 国家统计局API需要申请key，这里提供接口框架
    实际使用时可替换为真实API或本地缓存数据
    """
    
    BASE_URL = "https://data.stats.gov.cn"
    
    API_ENDPOINTS = {
        'gdp': '/easyqueryapi/sj/ndsj/',
        'cpi': '/easyqueryapi/sj/cpi/',
        'employment': '/easyqueryapi/sj/employment/'
    }
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        self.api_key = api_key or os.environ.get('NBS_API_KEY', '')
        self.use_cache = use_cache
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_path(self, data_type: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f'{data_type}_cache.json')
    
    def _load_from_cache(self, data_type: str) -> Optional[Dict]:
        """从缓存加载数据"""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(data_type)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                cache_time = cache_data.get('timestamp', 0)
                if time.time() - cache_time < 86400:
                    logger.info(f"从缓存加载 {data_type} 数据")
                    return cache_data.get('data')
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
        
        return None
    
    def _save_to_cache(self, data_type: str, data: Dict):
        """保存数据到缓存"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(data_type)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def fetch_gdp(self) -> Dict[str, Any]:
        """获取GDP数据"""
        cached = self._load_from_cache('gdp')
        if cached:
            return cached
        
        if not self.api_key:
            logger.warning("未配置API密钥，使用模拟数据")
            return self._get_mock_gdp_data()
        
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.API_ENDPOINTS['gdp']}",
                params={'apiKey': self.api_key},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self._save_to_cache('gdp', data)
                return data
        except Exception as e:
            logger.error(f"获取GDP数据失败: {e}")
        
        return self._get_mock_gdp_data()
    
    def fetch_cpi(self) -> Dict[str, Any]:
        """获取CPI数据"""
        cached = self._load_from_cache('cpi')
        if cached:
            return cached
        
        if not self.api_key:
            logger.warning("未配置API密钥，使用模拟数据")
            return self._get_mock_cpi_data()
        
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.API_ENDPOINTS['cpi']}",
                params={'apiKey': self.api_key},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self._save_to_cache('cpi', data)
                return data
        except Exception as e:
            logger.error(f"获取CPI数据失败: {e}")
        
        return self._get_mock_cpi_data()
    
    def fetch_employment(self) -> Dict[str, Any]:
        """获取就业数据"""
        cached = self._load_from_cache('employment')
        if cached:
            return cached
        
        if not self.api_key:
            logger.warning("未配置API密钥，使用模拟数据")
            return self._get_mock_employment_data()
        
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.API_ENDPOINTS['employment']}",
                params={'apiKey': self.api_key},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self._save_to_cache('employment', data)
                return data
        except Exception as e:
            logger.error(f"获取就业数据失败: {e}")
        
        return self._get_mock_employment_data()
    
    def _get_mock_gdp_data(self) -> Dict[str, Any]:
        """获取模拟GDP数据"""
        return {
            'source': 'mock',
            'data': {
                'years': list(range(2015, 2025)),
                'quarterly': [
                    {'year': 2024, 'quarter': q, 'value': round(30 + q * 2 + (q ** 0.5), 1)}
                    for q in range(1, 5)
                ],
                'annual': {
                    '2024': 126.1,
                    '2023': 121.0,
                    '2022': 121.0,
                    '2021': 110.4,
                    '2020': 101.6
                },
                'growth_rate': {
                    '2024': 4.2,
                    '2023': 3.0,
                    '2022': 9.5,
                    '2021': 8.4,
                    '2020': 2.2
                }
            },
            'description': '国内生产总值(亿元)',
            'updated': datetime.now().isoformat()
        }
    
    def _get_mock_cpi_data(self) -> Dict[str, Any]:
        """获取模拟CPI数据"""
        return {
            'source': 'mock',
            'data': {
                'years': list(range(2015, 2025)),
                'annual': {
                    '2024': 101.5,
                    '2023': 102.0,
                    '2022': 102.9,
                    '2021': 100.9,
                    '2020': 102.5
                },
                'monthly': {
                    '2024': [101.8, 101.2, 101.0, 101.5, 101.8, 101.5, 101.4, 101.8, 101.5, 101.5, 101.2, 101.2]
                }
            },
            'description': '居民消费价格指数(上年=100)',
            'updated': datetime.now().isoformat()
        }
    
    def _get_mock_employment_data(self) -> Dict[str, Any]:
        """获取模拟就业数据"""
        return {
            'source': 'mock',
            'data': {
                'years': list(range(2015, 2025)),
                'urban_employment': {
                    '2024': 625.0,
                    '2023': 613.1,
                    '2022': 587.3,
                    '2021': 542.3,
                    '2020': 514.1
                },
                'unemployment_rate': {
                    '2024': 4.0,
                    '2023': 4.2,
                    '2022': 3.9,
                    '2021': 4.4,
                    '2020': 5.3
                }
            },
            'description': '城镇就业人员数(万人)/城镇调查失业率(%)',
            'updated': datetime.now().isoformat()
        }


class MockDataSource(DataSourceBase):
    """本地模拟数据源"""
    
    def fetch_gdp(self) -> Dict[str, Any]:
        return NBSDataSource()._get_mock_gdp_data()
    
    def fetch_cpi(self) -> Dict[str, Any]:
        return NBSDataSource()._get_mock_cpi_data()
    
    def fetch_employment(self) -> Dict[str, Any]:
        return NBSDataSource()._get_mock_employment_data()


def get_data_source(source_type: str = 'auto') -> DataSourceBase:
    """获取数据源实例
    
    Args:
        source_type: 数据源类型 ('nbs', 'mock', 'auto')
    
    Returns:
        数据源实例
    """
    if source_type == 'nbs':
        return NBSDataSource()
    elif source_type == 'mock':
        return MockDataSource()
    else:
        api_key = os.environ.get('NBS_API_KEY', '')
        if api_key:
            return NBSDataSource()
        return MockDataSource()


if __name__ == '__main__':
    source = get_data_source()
    print("测试数据源...")
    print(f"\nGDP数据: {source.fetch_gdp()}")
    print(f"\nCPI数据: {source.fetch_cpi()}")
    print(f"\n就业数据: {source.fetch_employment()}")
