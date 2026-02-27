"""
缓存管理模块
实现基于文件系统的缓存机制，支持过期时间设置
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import logging

# 导入配置
import sys
sys.path.append('..')
from config import Config, PathConfig


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: Optional[Path] = None, expire_hours: int = 24):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            expire_hours: 缓存过期时间（小时）
        """
        self.cache_dir = cache_dir or PathConfig.CACHE_DIR
        self.expire_hours = expire_hours
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """
        生成缓存键（MD5哈希）
        
        Args:
            key: 原始键
        
        Returns:
            哈希后的键
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
        
        Returns:
            缓存文件路径
        """
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.cache"
    
    def _get_meta_path(self, key: str) -> Path:
        """
        获取缓存元数据文件路径
        
        Args:
            key: 缓存键
        
        Returns:
            元数据文件路径
        """
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
        
        Returns:
            缓存的数据，如果不存在或已过期则返回 None
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        # 检查缓存是否存在
        if not cache_path.exists() or not meta_path.exists():
            return None
        
        try:
            # 读取元数据
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # 检查是否过期
            cached_time = datetime.fromisoformat(meta['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.expire_hours):
                self.logger.info(f"缓存已过期: {key}")
                self.delete(key)
                return None
            
            # 读取缓存数据
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.debug(f"缓存命中: {key}")
            return data
        
        except Exception as e:
            self.logger.error(f"读取缓存失败: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, data: Any, metadata: Optional[Dict] = None):
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            metadata: 可选的元数据
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        try:
            # 保存缓存数据
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # 保存元数据
            meta = {
                'key': key,
                'timestamp': datetime.now().isoformat(),
                'expire_hours': self.expire_hours,
                'metadata': metadata or {}
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"缓存已保存: {key}")
        
        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")
    
    def delete(self, key: str):
        """
        删除缓存
        
        Args:
            key: 缓存键
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            self.logger.debug(f"缓存已删除: {key}")
        except Exception as e:
            self.logger.error(f"删除缓存失败: {e}")
    
    def clear_expired(self):
        """清除所有过期的缓存"""
        count = 0
        
        for meta_file in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                cached_time = datetime.fromisoformat(meta['timestamp'])
                expire_hours = meta.get('expire_hours', self.expire_hours)
                
                if datetime.now() - cached_time > timedelta(hours=expire_hours):
                    key = meta['key']
                    self.delete(key)
                    count += 1
            
            except Exception as e:
                self.logger.error(f"清理缓存失败: {meta_file} - {e}")
        
        self.logger.info(f"已清理 {count} 个过期缓存")
        return count
    
    def clear_all(self):
        """清除所有缓存"""
        count = 0
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                self.logger.error(f"删除缓存文件失败: {cache_file} - {e}")
        
        for meta_file in self.cache_dir.glob("*.meta"):
            try:
                meta_file.unlink()
            except Exception as e:
                self.logger.error(f"删除元数据文件失败: {meta_file} - {e}")
        
        self.logger.info(f"已清理 {count} 个缓存文件")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        meta_files = list(self.cache_dir.glob("*.meta"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'total_count': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'expire_hours': self.expire_hours
        }


# ==================== 装饰器：自动缓存 ====================

def cache_result(cache_manager: CacheManager, key_prefix: str = ""):
    """
    缓存函数结果的装饰器
    
    Args:
        cache_manager: 缓存管理器实例
        key_prefix: 缓存键前缀
    
    Example:
        @cache_result(cache_manager, "gdp_data")
        def fetch_gdp_data():
            # 复杂的数据获取逻辑
            return data
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}_{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # 尝试从缓存获取
            cached_data = cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 保存到缓存
            cache_manager.set(cache_key, result)
            
            return result
        return wrapper
    return decorator
