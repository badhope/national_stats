"""
统计分析模块
提供描述性统计、相关性分析、回归分析、景气指数计算等功能
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime
import logging
import warnings

# 导入统计建模库
import statsmodels.api as sm
from scipy import stats

# 导入配置和模型
import sys
sys.path.append('..')
from config import Config, IndicatorLibrary
from models.time_series import MacroTimeSeries


class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self):
        """初始化统计分析器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config.analysis
    
    # ==================== 1. 基础统计分析 ====================
    
    def descriptive_stats(self, ts: MacroTimeSeries) -> pd.DataFrame:
        """
        生成描述性统计表
        
        Args:
            ts: 时间序列对象
        
        Returns:
            包含统计指标的 DataFrame
        """
        data = ts.values
        
        stats_dict = {
            '指标名称': ts.name,
            '观测数': len(data),
            '均值': data.mean(),
            '标准差': data.std(),
            '最小值': data.min(),
            '25%分位数': data.quantile(0.25),
            '中位数': data.median(),
            '75%分位数': data.quantile(0.75),
            '最大值': data.max(),
            '偏度': data.skew(),
            '峰度': data.kurtosis(),
            '变异系数': data.std() / data.mean() if data.mean() != 0 else np.nan
        }
        
        return pd.DataFrame([stats_dict]).T.rename(columns={0: '统计值'})
    
    def calculate_growth_rates(self, ts: MacroTimeSeries, 
                               periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算增长率（同比、环比、累计同比）
        
        Args:
            ts: 时间序列对象
            periods: 周期数，默认自动根据频率推断
        
        Returns:
            包含原值和增长率的 DataFrame
        """
        df = ts.data.copy()
        
        # 自动推断周期
        if periods is None:
            freq = ts.meta.indicator.frequency
            periods = 12 if freq == 'monthly' else 4 if freq == 'quarterly' else 1
        
        # 同比增长率
        df['yoy'] = df['value'].pct_change(periods=periods) * 100
        
        # 环比增长率
        df['mom'] = df['value'].pct_change(periods=1) * 100
        
        # 移动平均
        df['ma_3'] = df['value'].rolling(window=3).mean()
        df['ma_12'] = df['value'].rolling(window=periods).mean()
        
        return df
    
    def standardize(self, ts: MacroTimeSeries, method: str = 'zscore') -> pd.Series:
        """
        数据标准化
        
        Args:
            ts: 时间序列对象
            method: 标准化方法 ('zscore' 或 'minmax')
        
        Returns:
            标准化后的序列
        """
        if method == 'zscore':
            return (ts.values - ts.values.mean()) / ts.values.std()
        elif method == 'minmax':
            return (ts.values - ts.values.min()) / (ts.values.max() - ts.values.min())
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    # ==================== 2. 相关性分析 ====================
    
    def correlation_matrix(self, ts_dict: Dict[str, MacroTimeSeries], 
                          method: str = 'pearson',
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        计算多指标相关性矩阵
        
        Args:
            ts_dict: 指标代码到时间序列的映射
            method: 相关性计算方法
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            相关性矩阵 DataFrame
        """
        # 合并数据
        df_list = []
        for code, ts in ts_dict.items():
            temp_df = ts.data[['value']].rename(columns={'value': code})
            df_list.append(temp_df)
        
        combined_df = pd.concat(df_list, axis=1)
        
        # 日期筛选
        if start_date:
            combined_df = combined_df[combined_df.index >= start_date]
        if end_date:
            combined_df = combined_df[combined_df.index <= end_date]
        
        # 计算相关性矩阵
        return combined_df.corr(method=method)
    
    def correlation_test(self, ts1: MacroTimeSeries, ts2: MacroTimeSeries) -> Dict[str, Any]:
        """
        两个指标的相关性检验
        
        Args:
            ts1: 第一个时间序列
            ts2: 第二个时间序列
        
        Returns:
            包含相关系数和P值的字典
        """
        # 对齐数据
        df = pd.merge(
            ts1.data[['value']], 
            ts2.data[['value']], 
            left_index=True, 
            right_index=True, 
            how='inner',
            suffixes=('_1', '_2')
        )
        
        if len(df) < 3:
            return {'error': '数据点不足，无法计算相关性'}
        
        # Pearson 相关系数
        corr, p_value = stats.pearsonr(df['value_1'], df['value_2'])
        
        # Spearman 相关系数
        spearman_corr, spearman_p = stats.spearmanr(df['value_1'], df['value_2'])
        
        return {
            'pearson_correlation': corr,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'sample_size': len(df),
            'interpretation': self._interpret_correlation(corr, p_value)
        }
    
    def _interpret_correlation(self, corr: float, p_value: float) -> str:
        """解释相关性强度"""
        if p_value > 0.05:
            return "相关性不显著 (p > 0.05)"
        
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            strength = "极强相关"
        elif abs_corr >= 0.6:
            strength = "强相关"
        elif abs_corr >= 0.4:
            strength = "中等相关"
        elif abs_corr >= 0.2:
            strength = "弱相关"
        else:
            strength = "极弱相关或无相关"
        
        direction = "正相关" if corr > 0 else "负相关"
        return f"{strength} ({direction})"
    
    # ==================== 3. 回归分析 ====================
    
    def ols_regression(self, y: MacroTimeSeries, x_list: List[MacroTimeSeries],
                      add_constant: bool = True) -> Dict[str, Any]:
        """
        多元线性回归分析
        
        Args:
            y: 因变量
            x_list: 自变量列表
            add_constant: 是否添加常数项
        
        Returns:
            回归结果字典
        """
        # 准备数据
        df = y.data[['value']].rename(columns={'value': 'y'})
        
        for i, ts in enumerate(x_list):
            col_name = ts.meta.indicator.code
            temp_df = ts.data[['value']].rename(columns={'value': col_name})
            df = df.join(temp_df, how='inner')
        
        df = df.dropna()
        
        if len(df) < len(x_list) + 2:
            return {'error': '有效数据点不足，无法进行回归分析'}
        
        # 构建回归模型
        X = df[[ts.meta.indicator.code for ts in x_list]]
        if add_constant:
            X = sm.add_constant(X)
        
        y_data = df['y']
        
        # 拟合模型
        try:
            model = sm.OLS(y_data, X).fit()
            
            return {
                'rsquared': model.rsquared,
                'rsquared_adj': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic,
                'coefficients': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                't_values': model.tvalues.to_dict(),
                'summary': model.summary().as_text(),
                'residuals': model.resid,
                'fitted_values': model.fittedvalues,
                'sample_size': len(df)
            }
        except Exception as e:
            self.logger.error(f"回归分析失败: {e}")
            return {'error': str(e)}
    
    # ==================== 4. 景气指数计算 ====================
    
    def calculate_diffusion_index(self, ts_dict: Dict[str, MacroTimeSeries],
                                  period: int = 1) -> pd.DataFrame:
        """
        计算扩散指数
        扩散指数 = (上升指标数 + 0.5 * 持平指标数) / 总指标数 * 100
        
        Args:
            ts_dict: 指标字典
            period: 比较周期（1表示环比，12表示同比）
        
        Returns:
            扩散指数时间序列
        """
        # 计算每个指标的变化方向
        direction_df = pd.DataFrame()
        
        for code, ts in ts_dict.items():
            temp_df = ts.data[['value']].rename(columns={'value': code})
            direction_df = direction_df.join(temp_df, how='outer')
        
        # 计算变化
        change_df = direction_df.diff(period)
        
        # 计算扩散指数
        results = []
        for date, row in change_df.iterrows():
            valid_values = row.dropna()
            if len(valid_values) == 0:
                continue
            
            up = (valid_values > 0).sum()
            down = (valid_values < 0).sum()
            unchanged = (valid_values == 0).sum()
            total = len(valid_values)
            
            di = (up + 0.5 * unchanged) / total * 100
            
            results.append({
                'date': date,
                'diffusion_index': di,
                'up_count': up,
                'down_count': down,
                'unchanged_count': unchanged
            })
        
        return pd.DataFrame(results).set_index('date')
    
    def calculate_composite_index(self, ts_dict: Dict[str, MacroTimeSeries],
                                  weights: Optional[Dict[str, float]] = None,
                                  base_period: Optional[str] = None) -> pd.DataFrame:
        """
        计算合成指数
        
        Args:
            ts_dict: 指标字典
            weights: 权重字典，如果为None则等权
            base_period: 基期，默认为数据起始期
        
        Returns:
            合成指数时间序列
        """
        if weights is None:
            weights = {code: 1.0/len(ts_dict) for code in ts_dict.keys()}
        
        # 标准化各指标
        normalized_df = pd.DataFrame()
        
        for code, ts in ts_dict.items():
            # 计算对称变化率
            temp_df = ts.data[['value']].rename(columns={'value': code})
            
            # 对称变化率：(X_t - X_{t-1}) / ((X_t + X_{t-1}) / 2)
            temp_df[f'{code}_scr'] = (temp_df[code] - temp_df[code].shift(1)) / \
                                    ((temp_df[code] + temp_df[code].shift(1)) / 2)
            
            # 标准化
            mean_scr = temp_df[f'{code}_scr'].mean()
            std_scr = temp_df[f'{code}_scr'].std()
            temp_df[f'{code}_normalized'] = temp_df[f'{code}_scr'] / std_scr if std_scr != 0 else 0
            
            normalized_df = normalized_df.join(temp_df[[f'{code}_normalized']], how='outer')
        
        # 计算加权平均
        normalized_df['composite'] = 0
        for code in ts_dict.keys():
            col = f'{code}_normalized'
            if col in normalized_df.columns:
                normalized_df['composite'] += normalized_df[col].fillna(0) * weights.get(code, 0)
        
        # 转换为指数形式（以基期为100）
        if base_period:
            base_value = normalized_df.loc[base_period, 'composite']
        else:
            base_value = normalized_df['composite'].iloc[0]
        
        normalized_df['composite_index'] = 100 * normalized_df['composite'] / base_value
        
        return normalized_df[['composite_index']].dropna()
    
    # ==================== 5. 时间序列分析 ====================
    
    def adf_test(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """
        ADF 单位根检验（平稳性检验）
        
        Args:
            ts: 时间序列对象
        
        Returns:
            检验结果字典
        """
        from statsmodels.tsa.stattools import adfuller
        
        values = ts.values.dropna()
        
        if len(values) < 20:
            return {'error': '数据点不足，无法进行ADF检验'}
        
        try:
            result = adfuller(values, autolag='AIC')
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'used_lag': result[2],
                'n_obs': result[3],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'interpretation': '序列平稳' if result[1] < 0.05 else '序列非平稳'
            }
        except Exception as e:
            self.logger.error(f"ADF检验失败: {e}")
            return {'error': str(e)}
    
    def granger_causality(self, ts1: MacroTimeSeries, ts2: MacroTimeSeries,
                         max_lag: int = 4) -> Dict[str, Any]:
        """
        格兰杰因果检验
        
        Args:
            ts1: 时间序列1（作为因）
            ts2: 时间序列2（作为果）
            max_lag: 最大滞后期
        
        Returns:
            检验结果
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # 准备数据
        df = pd.DataFrame({
            'y': ts2.values,
            'x': ts1.values
        }).dropna()
        
        if len(df) < max_lag * 3:
            return {'error': '数据点不足'}
        
        try:
            result = grangercausalitytests(df[['y', 'x']], maxlag=max_lag, verbose=False)
            
            results_dict = {}
            for lag in range(1, max_lag + 1):
                test_result = result[lag][0]  # 取第一个检验结果
                results_dict[f'lag_{lag}'] = {
                    'ssr_ftest': test_result['ssr_ftest'][1],  # p-value
                    'ssr_chi2test': test_result['ssr_chi2test'][1],
                }
            
            return {
                'test_results': results_dict,
                'conclusion': '存在格兰杰因果关系' if any(r['ssr_ftest'] < 0.05 for r in results_dict.values()) else '不存在格兰杰因果关系'
            }
        except Exception as e:
            self.logger.error(f"格兰杰因果检验失败: {e}")
            return {'error': str(e)}


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from core.data_manager import DataManager
    
    # 初始化
    analyzer = StatisticalAnalyzer()
    dm = DataManager()
    
    # 获取数据
    gdp_ts = dm.fetch("gdp")
    cpi_ts = dm.fetch("cpi")
    
    if gdp_ts and cpi_ts:
        # 1. 描述性统计
        print("=== GDP 描述性统计 ===")
        print(analyzer.descriptive_stats(gdp_ts))
        
        # 2. 增长率计算
        print("\n=== GDP 增长率 ===")
        growth_df = analyzer.calculate_growth_rates(gdp_ts)
        print(growth_df.tail())
        
        # 3. 相关性分析
        print("\n=== GDP 与 CPI 相关性 ===")
        corr_result = analyzer.correlation_test(gdp_ts, cpi_ts)
        print(corr_result['interpretation'])
        
        # 4. ADF 检验
        print("\n=== GDP 平稳性检验 ===")
        adf_result = analyzer.adf_test(gdp_ts)
        print(adf_result['interpretation'])
    
    dm.close()
