"""
奥肯定律模型
研究GDP增长与失业率变动之间的关系
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from scipy import stats

# 导入基础模块
import sys
sys.path.append('../..')
from config import Config
from models.time_series import MacroTimeSeries


class OkunLawModel:
    """奥肯定律模型"""
    
    def __init__(self):
        """初始化奥肯定律模型"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.coefficients = None
    
    def estimate(self, 
                gdp_ts: MacroTimeSeries,
                unemployment_ts: MacroTimeSeries,
                method: str = 'diff') -> Dict[str, Any]:
        """
        估计奥肯定律系数
        
        奥肯定律形式：
        1. 差分形式：Δu = a + b * g_Y
        2. 水平形式：u - u* = a + b * (Y - Y*)/Y*
        
        Args:
            gdp_ts: GDP时间序列
            unemployment_ts: 失业率时间序列
            method: 估计方法
        
        Returns:
            估计结果字典
        """
        # 计算GDP增长率
        gdp_growth = self._calculate_growth_rate(gdp_ts)
        
        # 计算失业率变化
        unemployment_change = unemployment_ts.values.diff()
        
        # 对齐数据
        df = pd.DataFrame({
            'gdp_growth': gdp_growth,
            'unemployment_change': unemployment_change
        }).dropna()
        
        if len(df) < 10:
            return {'error': '有效数据点不足，需要至少10个观测值'}
        
        # 执行回归
        X = df['gdp_growth'].values
        y = df['unemployment_change'].values
        
        # 添加常数项
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            # 使用最小二乘法
            model = np.linalg.lstsq(X_with_const, y, rcond=None)
            coefficients = model[0]
            intercept, slope = coefficients[0], coefficients[1]
            
            # 计算R²
            y_pred = intercept + slope * X
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # 计算标准误差
            n = len(df)
            mse = ss_res / (n - 2)
            se_slope = np.sqrt(mse / np.sum((X - np.mean(X)) ** 2))
            
            # t检验
            t_stat = slope / se_slope
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            # 奥肯定律系数（绝对值）
            okun_coefficient = abs(slope)
            
            self.coefficients = {
                'intercept': intercept,
                'slope': slope,
                'okun_coefficient': okun_coefficient
            }
            
            return {
                'okun_coefficient': okun_coefficient,
                'intercept': intercept,
                'slope': slope,
                'r_squared': r_squared,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'sample_size': n,
                'interpretation': self._interpret_result(okun_coefficient, p_value),
                'equation': f'Δu = {intercept:.3f} + {slope:.3f} * g_Y',
                'detailed_data': df,
                'predictions': y_pred
            }
        
        except Exception as e:
            self.logger.error(f"奥肯定律估计失败: {e}")
            return {'error': str(e)}
    
    def _calculate_growth_rate(self, ts: MacroTimeSeries) -> pd.Series:
        """计算同比增长率"""
        freq = ts.meta.indicator.frequency
        periods = 12 if freq == 'monthly' else 4 if freq == 'quarterly' else 1
        return ts.values.pct_change(periods=periods) * 100
    
    def _interpret_result(self, coefficient: float, p_value: float) -> str:
        """解释结果"""
        if p_value > 0.05:
            return "奥肯定律关系不显著 (p > 0.05)"
        
        if coefficient < 0.1:
            strength = "较弱"
        elif coefficient < 0.3:
            strength = "中等"
        else:
            strength = "较强"
        
        return f"奥肯定律关系显著，系数为{coefficient:.3f}，表示GDP增长1个百分点，失业率下降约{coefficient:.2f}个百分点，关系{strength}"
    
    def predict_unemployment_change(self, gdp_growth: float) -> float:
        """
        根据GDP增长率预测失业率变化
        
        Args:
            gdp_growth: GDP增长率（%）
        
        Returns:
            预测的失业率变化（百分点）
        """
        if self.coefficients is None:
            raise ValueError("模型尚未估计，请先调用estimate方法")
        
        return self.coefficients['intercept'] + self.coefficients['slope'] * gdp_growth
    
    def visualize(self, result: Dict[str, Any], 
                 figsize: Tuple[int, int] = (10, 6)):
        """可视化奥肯定律关系"""
        import matplotlib.pyplot as plt
        
        if 'error' in result:
            print("无法可视化：估计失败")
            return None
        
        df = result['detailed_data']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 散点图
        ax.scatter(df['gdp_growth'], df['unemployment_change'], 
                  alpha=0.6, s=50, color='#3498DB', label='实际数据')
        
        # 回归线
        x_line = np.linspace(df['gdp_growth'].min(), df['gdp_growth'].max(), 100)
        y_line = result['intercept'] + result['slope'] * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f"拟合线 (R²={result['r_squared']:.3f})")
        
        # 添加零线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_xlabel('GDP增长率 (%)', fontsize=12)
        ax.set_ylabel('失业率变化 (百分点)', fontsize=12)
        ax.set_title('奥肯定律关系', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加方程文本
        ax.text(0.05, 0.95, result['equation'], transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 注意：中国官方失业率数据可能不完全符合奥肯定律假设
    # 这里使用示例数据演示
    
    print("奥肯定律模型示例")
    print("=" * 50)
    print("注意：奥肯定律在中国经济的适用性需要谨慎分析")
    print("主要因为：")
    print("1. 中国失业率统计口径与西方国家不同")
    print("2. 存在大量隐性就业和农村剩余劳动力")
    print("3. 劳动力市场结构性特征明显")
