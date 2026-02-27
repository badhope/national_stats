"""
经济增长核算模型
将经济增长分解为资本、劳动和全要素生产率（TFP）的贡献
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

# 导入基础模块
import sys
sys.path.append('../..')
from config import Config
from models.time_series import MacroTimeSeries


class GrowthAccountingModel:
    """增长核算模型"""
    
    def __init__(self, capital_share: float = 0.4):
        """
        初始化增长核算模型
        
        Args:
            capital_share: 资本产出弹性（α），通常取值0.3-0.5，默认0.4
        """
        self.capital_share = capital_share
        self.labor_share = 1 - capital_share
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate(self, 
                 gdp_ts: MacroTimeSeries,
                 capital_ts: Optional[MacroTimeSeries] = None,
                 labor_ts: Optional[MacroTimeSeries] = None) -> Dict[str, Any]:
        """
        执行增长核算
        
        基于索洛增长模型：Y = A * K^α * L^(1-α)
        增长率分解：g_Y = g_A + α * g_K + (1-α) * g_L
        
        Args:
            gdp_ts: GDP时间序列
            capital_ts: 资本存量时间序列（可选）
            labor_ts: 劳动力时间序列（可选）
        
        Returns:
            包含核算结果的字典
        """
        # 计算GDP增长率
        gdp_growth = self._calculate_growth_rate(gdp_ts)
        
        # 如果缺少资本或劳动数据，使用简化估算
        if capital_ts is None or labor_ts is None:
            return self._simplified_accounting(gdp_ts, gdp_growth)
        
        # 完整核算
        capital_growth = self._calculate_growth_rate(capital_ts)
        labor_growth = self._calculate_growth_rate(labor_ts)
        
        # 对齐数据时间
        df = pd.DataFrame({
            'gdp_growth': gdp_growth,
            'capital_growth': capital_growth,
            'labor_growth': labor_growth
        }).dropna()
        
        # 计算各要素贡献
        df['capital_contribution'] = self.capital_share * df['capital_growth']
        df['labor_contribution'] = self.labor_share * df['labor_growth']
        df['tfp_growth'] = df['gdp_growth'] - df['capital_contribution'] - df['labor_contribution']
        
        # 计算贡献率
        df['capital_contribution_rate'] = df['capital_contribution'] / df['gdp_growth'] * 100
        df['labor_contribution_rate'] = df['labor_contribution'] / df['gdp_growth'] * 100
        df['tfp_contribution_rate'] = df['tfp_growth'] / df['gdp_growth'] * 100
        
        # 汇总统计
        summary = {
            'period': f"{df.index.min().year} - {df.index.max().year}",
            'avg_gdp_growth': df['gdp_growth'].mean(),
            'avg_capital_contribution': df['capital_contribution'].mean(),
            'avg_labor_contribution': df['labor_contribution'].mean(),
            'avg_tfp_growth': df['tfp_growth'].mean(),
            'capital_share': self.capital_share,
            'detailed_data': df
        }
        
        return summary
    
    def _simplified_accounting(self, gdp_ts: MacroTimeSeries, 
                              gdp_growth: pd.Series) -> Dict[str, Any]:
        """
        简化核算（当缺少要素数据时）
        假设资本增长与GDP增长正相关，劳动增长假设为固定值
        """
        self.logger.warning("缺少资本或劳动数据，使用简化估算")
        
        # 假设资本增长率为GDP增长率的1.2倍（经验值）
        capital_growth = gdp_growth * 1.2
        
        # 假设劳动增长率为0.5%（近似人口增长率）
        labor_growth = pd.Series(0.5, index=gdp_growth.index)
        
        df = pd.DataFrame({
            'gdp_growth': gdp_growth,
            'capital_growth': capital_growth,
            'labor_growth': labor_growth
        })
        
        df['capital_contribution'] = self.capital_share * df['capital_growth']
        df['labor_contribution'] = self.labor_share * df['labor_growth']
        df['tfp_growth'] = df['gdp_growth'] - df['capital_contribution'] - df['labor_contribution']
        
        summary = {
            'period': f"{df.index.min().year} - {df.index.max().year}",
            'avg_gdp_growth': df['gdp_growth'].mean(),
            'avg_capital_contribution': df['capital_contribution'].mean(),
            'avg_labor_contribution': df['labor_contribution'].mean(),
            'avg_tfp_growth': df['tfp_growth'].mean(),
            'capital_share': self.capital_share,
            'note': '简化估算结果，建议补充资本和劳动数据',
            'detailed_data': df
        }
        
        return summary
    
    def _calculate_growth_rate(self, ts: MacroTimeSeries) -> pd.Series:
        """计算年度增长率"""
        # 如果是季度数据，转换为年度
        if ts.meta.indicator.frequency == 'quarterly':
            ts_annual = ts.resample('Y', agg_func='sum')
            values = ts_annual.values
        else:
            values = ts.values
        
        # 计算增长率
        growth = values.pct_change() * 100
        return growth
    
    def visualize_contributions(self, result: Dict[str, Any], 
                               figsize: Tuple[int, int] = (12, 6)):
        """可视化增长贡献（堆叠柱状图）"""
        import matplotlib.pyplot as plt
        
        df = result['detailed_data']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 准备数据
        periods = [str(idx.year) for idx in df.index]
        capital = df['capital_contribution'].values
        labor = df['labor_contribution'].values
        tfp = df['tfp_growth'].values
        
        # 绘制堆叠柱状图
        ax.bar(periods, capital, label='资本贡献', color='#3498DB')
        ax.bar(periods, labor, bottom=capital, label='劳动贡献', color='#2ECC71')
        ax.bar(periods, tfp, bottom=capital+labor, label='TFP贡献', color='#E74C3C')
        
        # 添加GDP增长率折线
        ax2 = ax.twinx()
        ax2.plot(periods, df['gdp_growth'].values, 'k--', marker='o', 
                label='GDP增长率', linewidth=2)
        
        ax.set_xlabel('年份')
        ax.set_ylabel('贡献百分点 (%)')
        ax.set_title('经济增长来源分解', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from core.data_manager import DataManager
    
    # 初始化
    model = GrowthAccountingModel(capital_share=0.4)
    dm = DataManager()
    
    # 获取数据
    gdp_ts = dm.fetch("gdp")
    
    if gdp_ts:
        # 执行简化核算
        result = model.calculate(gdp_ts)
        
        print(f"=== 增长核算结果 ({result['period']}) ===")
        print(f"平均GDP增长率: {result['avg_gdp_growth']:.2f}%")
        print(f"资本贡献: {result['avg_capital_contribution']:.2f}%")
        print(f"劳动贡献: {result['avg_labor_contribution']:.2f}%")
        print(f"TFP贡献: {result['avg_tfp_growth']:.2f}%")
        
        # 可视化
        fig = model.visualize_contributions(result)
        plt.show()
    
    dm.close()
