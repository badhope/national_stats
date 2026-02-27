"""
数据库管理模块
使用 SQLite 存储时间序列数据，支持增删改查操作
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import json

# 导入配置和模型
import sys
sys.path.append('..')
from config import Config, PathConfig, IndicatorLibrary
from models.time_series import MacroTimeSeries, TimeSeriesMeta


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path or PathConfig.DATABASE_FILE
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库表
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 返回字典形式的行
        return conn
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建时间序列数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_code TEXT NOT NULL,
                date TEXT NOT NULL,
                value REAL,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(indicator_code, date)
            )
        ''')
        
        # 创建指标元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicator_metadata (
                indicator_code TEXT PRIMARY KEY,
                indicator_name TEXT,
                category TEXT,
                frequency TEXT,
                unit TEXT,
                source TEXT,
                description TEXT,
                is_leading INTEGER,
                is_coincident INTEGER,
                is_lagging INTEGER,
                weight REAL,
                last_updated TEXT,
                data_count INTEGER
            )
        ''')
        
        # 创建数据缓存表（用于缓存复杂查询结果）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key TEXT PRIMARY KEY,
                cache_data TEXT,
                created_at TEXT,
                expire_at TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_indicator_date 
            ON time_series(indicator_code, date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_indicator_code 
            ON time_series(indicator_code)
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("数据库初始化完成")
    
    def save_time_series(self, ts: MacroTimeSeries) -> int:
        """
        保存时间序列数据到数据库
        
        Args:
            ts: 时间序列对象
        
        Returns:
            插入/更新的记录数
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 准备数据
        data = ts.data.reset_index()
        indicator_code = ts.meta.indicator.code
        source = ts.meta.source
        
        count = 0
        now = datetime.now().isoformat()
        
        for _, row in data.iterrows():
            date = str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date'])
            value = row['value']
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO time_series 
                    (indicator_code, date, value, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (indicator_code, date, value, source, now, now))
                count += 1
            except Exception as e:
                self.logger.error(f"插入数据失败: {date} - {e}")
        
        # 更新指标元数据
        self._update_indicator_metadata(cursor, ts, count, now)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"保存了 {count} 条数据: {indicator_code}")
        return count
    
    def _update_indicator_metadata(self, cursor, ts: MacroTimeSeries, count: int, now: str):
        """更新指标元数据"""
        indicator = ts.meta.indicator
        
        cursor.execute('''
            INSERT OR REPLACE INTO indicator_metadata
            (indicator_code, indicator_name, category, frequency, unit, source,
             description, is_leading, is_coincident, is_lagging, weight,
             last_updated, data_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            indicator.code,
            indicator.name,
            indicator.category,
            indicator.frequency,
            indicator.unit,
            indicator.source,
            indicator.description,
            1 if indicator.is_leading else 0,
            1 if indicator.is_coincident else 0,
            1 if indicator.is_lagging else 0,
            indicator.weight,
            now,
            count
        ))
    
    def load_time_series(self, indicator_code: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[MacroTimeSeries]:
        """
        从数据库加载时间序列数据
        
        Args:
            indicator_code: 指标代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            时间序列对象，如果不存在则返回 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构建查询
        query = "SELECT date, value FROM time_series WHERE indicator_code = ?"
        params = [indicator_code]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return None
        
        # 转换为 DataFrame
        data = pd.DataFrame([dict(row) for row in rows])
        
        # 获取指标定义
        try:
            indicator = IndicatorLibrary.get_indicator(indicator_code)
        except ValueError:
            conn.close()
            return None
        
        # 创建元数据
        meta = TimeSeriesMeta(
            indicator=indicator,
            source="database",
            last_updated=datetime.now()
        )
        
        conn.close()
        
        return MacroTimeSeries(data, meta)
    
    def get_indicator_list(self) -> List[Dict[str, Any]]:
        """获取数据库中所有指标的列表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT indicator_code, indicator_name, category, frequency, unit,
                   last_updated, data_count
            FROM indicator_metadata
            ORDER BY category, indicator_code
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def delete_indicator_data(self, indicator_code: str) -> int:
        """
        删除指定指标的所有数据
        
        Args:
            indicator_code: 指标代码
        
        Returns:
            删除的记录数
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM time_series WHERE indicator_code = ?', (indicator_code,))
        cursor.execute('DELETE FROM indicator_metadata WHERE indicator_code = ?', (indicator_code,))
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        self.logger.info(f"删除了指标 {indicator_code} 的 {count} 条数据")
        return count
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 总记录数
        cursor.execute('SELECT COUNT(*) as count FROM time_series')
        total_records = cursor.fetchone()['count']
        
        # 指标数量
        cursor.execute('SELECT COUNT(*) as count FROM indicator_metadata')
        total_indicators = cursor.fetchone()['count']
        
        # 数据库文件大小
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        # 按类别统计
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM indicator_metadata
            GROUP BY category
        ''')
        by_category = {row['category']: row['count'] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_records': total_records,
            'total_indicators': total_indicators,
            'database_size_mb': db_size / (1024 * 1024),
            'by_category': by_category,
            'database_path': str(self.db_path)
        }
    
    def export_to_csv(self, indicator_code: str, output_path: Path) -> bool:
        """
        将指标数据导出为 CSV 文件
        
        Args:
            indicator_code: 指标代码
            output_path: 输出文件路径
        
        Returns:
            是否成功
        """
        ts = self.load_time_series(indicator_code)
        
        if ts is None:
            return False
        
        try:
            ts.data.to_csv(output_path, index=True, encoding='utf-8-sig')
            self.logger.info(f"已导出到: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"导出失败: {e}")
            return False


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建数据库管理器
    db = DatabaseManager()
    
    # 获取统计信息
    stats = db.get_database_stats()
    print(f"数据库统计: {stats}")
    
    # 获取指标列表
    indicators = db.get_indicator_list()
    print(f"\n已有指标: {len(indicators)} 个")
