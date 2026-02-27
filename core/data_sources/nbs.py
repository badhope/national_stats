"""
国家统计局数据源
从国家统计局官网获取宏观经济数据
"""

import pandas as pd
import json
import time  # 添加time模块导入
from typing import Optional, Dict, Any, List
from datetime import datetime

# 导入基类和模型
import sys
sys.path.append('../..')
from config import IndicatorLibrary, IndicatorDefinition
from models.time_series import MacroTimeSeries, TimeSeriesMeta, create_time_series
from .base import BaseDataSource


class NBSDataSource(BaseDataSource):
    """国家统计局数据源"""
    
    # 国家统计局 API 端点
    BASE_URL = "https://data.stats.gov.cn/easyquery.htm"
    
    # 指标代码映射到国家统计局数据库 ID
    # 这需要根据实际 API 进行映射
    INDICATOR_MAP = {
        # GDP
        "gdp": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A01"},
        "gdp_yoy": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0103"},
        
        # 工业
        "industrial_yoy": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0201"},
        
        # 价格
        "cpi": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0801"},
        "cpi_yoy": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0802"},
        "ppi": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0807"},
        
        # 投资
        "fixed_asset_investment_yoy": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0601"},
        
        # 消费
        "retail_sales_yoy": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A0501"},
        
        # PMI
        "pmi_manufacturing": {"dbcode": "fsyd", "rowcode": "zb", "colcode": "sj", "id": "A1901"},
    }
    
    def fetch(self, indicator_code: str, start_date: Optional[str] = None,
              end_date: Optional[str] = None, **kwargs) -> MacroTimeSeries:
        """
        获取数据
        
        Args:
            indicator_code: 指标代码（必须在 INDICATOR_MAP 中定义）
            start_date: 开始日期，格式 '2020-01'
            end_date: 结束日期
        """
        # 检查指标是否支持
        if indicator_code not in self.INDICATOR_MAP:
            raise ValueError(f"不支持的指标代码: {indicator_code}")
        
        # 获取映射参数
        params = self.INDICATOR_MAP[indicator_code]
        
        # 构建请求参数
        query_params = {
            "m": "QueryData",
            "dbcode": params["dbcode"],
            "rowcode": params["rowcode"],
            "colcode": params["colcode"],
            "wds": "[]",
            "dfwds": '[{"wdcode":"zb","valuecode":"' + params["id"] + '"}]',
            "k1": str(int(time.time() * 1000)),  # 时间戳
            "h": "1",
        }
        
        # 发送请求
        response = self._request(self.BASE_URL, params=query_params)
        
        if response is None:
            raise ConnectionError(f"获取数据失败: {indicator_code}")
        
        # 解析数据
        try:
            data = self._parse_response(response.json(), indicator_code)
        except Exception as e:
            self.logger.error(f"解析数据失败: {e}")
            raise ValueError(f"数据解析失败: {indicator_code}")
        
        # 创建时间序列对象
        ts = create_time_series(indicator_code, data, source="nbs")
        
        # 日期筛选
        if start_date or end_date:
            ts = ts.filter_date(start_date or "1900-01-01", end_date or "2100-12-31")
        
        return ts
    
    def _parse_response(self, response_data: Dict, indicator_code: str) -> pd.DataFrame:
        """
        解析 API 响应
        
        国家统计局 API 返回的数据结构复杂，需要逐层解析
        """
        records = []
        
        try:
            # 提取数据节点
            datanodes = response_data.get("returndata", {}).get("datanodes", [])
            
            for node in datanodes:
                # 提取时间维度
                wds = node.get("wds", [])
                time_str = None
                for wd in wds:
                    if wd.get("wdcode") == "sj":
                        time_str = wd.get("valuecode")
                        break
                
                # 提取数值
                value = node.get("data", {}).get("data")
                
                if time_str and value is not None:
                    # 转换时间格式
                    # 国家统计局格式可能是 "2020A" (年), "2020Q1" (季), "2020M01" (月)
                    date = self._parse_time_string(time_str)
                    
                    if date:
                        records.append({
                            'date': date,
                            'value': float(value)
                        })
        
        except Exception as e:
            self.logger.error(f"解析错误: {e}")
        
        # 创建 DataFrame
        df = pd.DataFrame(records)
        
        # 去重并排序
        if len(df) > 0:
            df = df.drop_duplicates(subset=['date']).sort_values('date')
        
        return df
    
    def _parse_time_string(self, time_str: str) -> Optional[str]:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串，如 "2020", "2020Q1", "2020M01"
        
        Returns:
            标准日期字符串，如 "2020-12-31", "2020-03-31", "2020-01-01"
        """
        try:
            if 'M' in time_str:
                # 月度数据: 2020M01 -> 2020-01-01
                year, month = time_str.replace('M', ' ').split()
                return f"{year}-{int(month):02d}-01"
            elif 'Q' in time_str:
                # 季度数据: 2020Q1 -> 2020-03-31 (季度末)
                year, quarter = time_str.replace('Q', ' ').split()
                quarter = int(quarter)
                month = quarter * 3
                return f"{year}-{month:02d}-01"
            elif 'A' in time_str:
                # 年度数据: 2020A -> 2020-12-31
                year = time_str.replace('A', '')
                return f"{year}-12-31"
            else:
                # 尝试直接解析
                return time_str
        except:
            return None
    
    def get_available_indicators(self) -> List[Dict[str, Any]]:
        """获取所有支持的指标列表"""
        indicators = []
        for code, params in self.INDICATOR_MAP.items():
            try:
                ind_def = IndicatorLibrary.get_indicator(code)
                indicators.append({
                    "code": code,
                    "name": ind_def.name,
                    "frequency": ind_def.frequency,
                    "unit": ind_def.unit
                })
            except:
                pass
        return indicators


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import time
    
    # 创建数据源
    with NBSDataSource() as nbs:
        # 获取 GDP 数据
        print("正在获取 GDP 数据...")
        gdp_ts = nbs.fetch("gdp")
        print(gdp_ts)
        print(gdp_ts.describe())
        
        # 获取 CPI 数据
        print("\n正在获取 CPI 数据...")
        cpi_ts = nbs.fetch("cpi")
        print(cpi_ts)
        
        # 计算同比增长率
        yoy = cpi_ts.yoy()
        print(f"CPI 同比增长: {yoy.iloc[-1]:.2f}%")