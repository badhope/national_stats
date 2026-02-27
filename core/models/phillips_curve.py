"""
菲利普斯曲线模型
研究失业率（或产出缺口）与通货膨胀率之间的关系
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from scipy import stats
from statsmodels.tsa.filters.hp_filter import hpfilter

# 导入基础模块
import sys
sys.path.append('../..')
from config import Config
from models.time_series import MacroTimeSeries


class PhillipsCurveModel:
    """菲利普斯曲线模型"""
    
    def __init__(self, model_type: str = 'expectations_augmented'):
        """
        初始化菲利普斯曲线模型
        
        Args:
            model_type: 模型类型
                - 'original': 原始菲利普斯曲线 (π = a + b*u)
                - 'expectations_augmented': 附加预期的菲利普斯曲线 (π = π_e - b*(u - u*))
                - 'new_keynesian': 新凯恩斯菲利普斯曲线
        """
        self.model_type = model_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self.coefficients = None
    
    def estimate(self, 
                inflation_ts: MacroTimeSeries,
                unemployment_ts: MacroTimeSeries,
                expected_inflation: str = 'adaptive') -> Dict[str, Any]:
        """
        估计菲利普斯曲线
        
        Args:
            inflation_ts: 通货膨胀率时间序列（通常是CPI）
            unemployment_ts: 失业率时间序列
            expected_inflation: 预期通胀类型
        
        Returns:
            估计结果字典
        """
        # 计算通胀率（如果是指数形式）
        inflation = self._process_inflation(inflation_ts)
        
        # 计算失业率
        unemployment = unemployment_ts.values
        
        # 对齐数据
        df = pd.DataFrame({
            'inflation': inflation,
            'unemployment': unemployment
        }).dropna()
        
        if len(df) < 15:
            return {'error': '有效数据点不足，需要至少15个观测值'}
        
        # 根据模型类型估计
        if self.model_type == 'original':
            return self._estimate_original(df)
        elif self.model_type == 'expectations_augmented':
            return self._estimate_expectations_augmented(df, expected_inflation)
        else:
            return {'error': f"不支持的模型类型: {self.model_type}"}
    
    def _process_inflation(self, ts: MacroTimeSeries) -> pd.Series:
        """处理通胀数据"""
        # 如果是同比指数（上年=100），转换为涨跌幅
        values = ts.values
        
        # 检查是否需要转换
        if values.mean() > 50:  # 可能是指数形式
            return values - 100
        
        return values
    
    def _estimate_original(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        估计原始菲利普斯曲线
        π_t = a + b * u_t
        """
        X = df['unemployment'].values
        y = df['inflation'].values
        
        # 添加常数项
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS回归
        model = np.linalg.lstsq(X_with_const, y, rcond=None)
        intercept, slope = model[0][0], model[0][1]
        
        # 计算统计量
        y_pred = intercept + slope * X
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return {
            'model_type': 'original',
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_squared,
            'equation': f'π = {intercept:.3f} + {slope:.3f} * u',
            'detailed_data': df,
            'predictions': y_pred,
            'interpretation': self._interpret_slope(slope)
        }
    
    def _estimate_expectations_augmented(self, df: pd.DataFrame, 
                                        expected_inflation: str) -> Dict[str, Any]:
        """
        估计附加预期的菲利普斯曲线
        π_t = π_e - β * (u_t - u*) + ε_t
        
        其中：
        - π_e: 预期通胀（自适应预期用滞后项）
        - u*: 自然失业率（使用HP滤波估算）
        """
        # 计算预期通胀
        if expected_inflation == 'adaptive':
            df['expected_inflation'] = df['inflation'].shift(1)
        else:
            # 简单假设预期通胀为历史平均
            df['expected_inflation'] = df['inflation'].mean()
        
        # 估算自然失业率（使用HP滤波）
        try:
            cycle, trend = hpfilter(df['unemployment'].dropna(), lamb=1600)
            df['natural_unemployment'] = trend
            df['unemployment_gap'] = df['unemployment'] - df['natural_unemployment']
        except:
            # 如果HP滤波失败，使用简单移动平均
            df['natural_unemployment'] = df['unemployment'].rolling(window=12, center=True).mean()
            df['unemployment_gap'] = df['unemployment'] - df['natural_unemployment']
        
        df = df.dropna()
        
        if len(df) < 10:
            return {'error': '估算自然失业率后有效数据不足'}
        
        # 回归：π_t - π_e = α + β * (u_t - u*)
        df['inflation_gap'] = df['inflation'] - df['expected_inflation']
        
        X = df['unemployment_gap'].values
        y = df['inflation_gap'].values
        
        X_with_const = np.column_stack([np.ones(len(X)), X])
        model = np.linalg.lstsq(X_with_const, y, rcond=None)
        intercept, slope = model[0][0], model[0][1]
        
        y_pred = intercept + slope * X
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        self.coefficients = {'intercept': intercept, 'slope': slope}
        
        return {
            'model_type': 'expectations_augmented',
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_squared,
            'natural_unemployment': df['natural_unemployment'].mean(),
            'equation': f'π = π_e + {intercept:.3f} - {abs(slope):.3f} * (u - u*)',
            'detailed_data': df,
            'predictions': y_pred,
            'interpretation': self._interpret_augmented(slope, intercept)
        }
    
    def _interpret_slope(self, slope: float) -> str:
        """解释原始菲利普斯曲线斜率"""
        if slope < 0:
            return f"负斜率 ({slope:.3f})，符合菲利普斯曲线预期：失业率越高，通胀越低"
        else:
            return f"正斜率 ({slope:.3f})，与传统菲利普斯曲线矛盾，可能存在滞胀或供给冲击"
    
    def _interpret_augmented(self, slope: float, intercept: float) -> str:
        """解释附加预期的菲利普斯曲线"""
        if slope < 0:
            return f"斜率显著为负 ({slope:.3f})，符合理论预期：失业率缺口对通胀有抑制作用"
        else:
            return f"斜率为正 ({slope:.3f})，短期权衡关系可能失效"
    
    def visualize(self, result: Dict[str, Any], 
                 figsize: Tuple[int, int] = (10, 6)):
        """可视化菲利普斯曲线"""
        import matplotlib.pyplot as plt
        
        if 'error' in result:
            print("无法可视化：估计失败")
            return None
        
        df = result['detailed_data']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 散点图
        scatter = ax.scatter(df['unemployment'], df['inflation'], 
                           c=range(len(df)), cmap='viridis', 
                           alpha=0.7, s=50, label='实际数据')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('时间序列')
        
        # 回归线
        x_line = np.linspace(df['unemployment'].min(), df['unemployment'].max(), 100)
        y_line = result['intercept'] + result['slope'] * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f"拟合线 (R²={result['r_squared']:.3f})")
        
        ax.set_xlabel('失业率 (%)', fontsize=12)
        ax.set_ylabel('通货膨胀率 (%)', fontsize=12)
        ax.set_title('菲利普斯曲线', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加方程
        ax.text(0.05, 0.95, result['equation'], transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("菲利普斯曲线模型")
    print("=" * 50)
    print("该模型用于分析失业率与通胀之间的权衡关系")
    print("\n注意事项：")
    print("1. 中国数据可能表现出与理论不同的特征")
    print("2. 供给侧冲击可能导致曲线移动")
    print("3. 通胀预期对曲线形态有重要影响")
