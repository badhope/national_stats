"""
数据源基类
定义所有数据源必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import requests
import time
import logging

# 导入配置和模型
import sys
sys.path.append('../..')
from config import Config, ScraperConfig
from models.time_series import MacroTimeSeries, TimeSeriesMeta


class BaseDataSource(ABC):
    """数据源抽象基类"""
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        初始化数据源
        
        Args:
            config: 爬虫配置，如果不传则使用全局配置
        """
        self.config = config or Config.scraper
        self.logger = logging.getLogger(self.__class__.__name__)
        self._session = None
    
    @property
    def session(self) -> requests.Session:
        """获取请求会话"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            })
            
            # 设置代理
            if self.config.proxies:
                self._session.proxies.update(self.config.proxies)
        
        return self._session
    
    @abstractmethod
    def fetch(self, indicator_code: str, start_date: Optional[str] = None, 
              end_date: Optional[str] = None, **kwargs) -> MacroTimeSeries:
        """
        获取数据（抽象方法，子类必须实现）
        
        Args:
            indicator_code: 指标代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        
        Returns:
            MacroTimeSeries 对象
        """
        pass
    
    def _request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """
        发送 HTTP 请求（带重试机制）
        
        Args:
            url: 请求 URL
            method: 请求方法
            **kwargs: requests 参数
        
        Returns:
            Response 对象或 None
        """
        for attempt in range(self.config.retry_times):
            try:
                # 添加延迟
                if attempt > 0:
                    time.sleep(self.config.retry_delay * attempt)
                
                # 发送请求
                if method.upper() == 'GET':
                    response = self.session.get(
                        url, 
                        timeout=self.config.timeout,
                        **kwargs
                    )
                else:
                    response = self.session.post(
                        url,
                        timeout=self.config.timeout,
                        **kwargs
                    )
                
                # 检查状态码
                if response.status_code == 200:
                    return response
                else:
                    self.logger.warning(f"请求失败 (状态码 {response.status_code}): {url}")
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"请求异常 (尝试 {attempt + 1}/{self.config.retry_times}): {e}")
        
        self.logger.error(f"请求最终失败: {url}")
        return None
    
    def close(self):
        """关闭会话"""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
