"""
数据分析与预测模块
使用统计模型和机器学习进行趋势预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class TrendPredictor:
    """趋势预测器"""
    
    def __init__(self):
        self.models = {}
        
    def linear_regression(self, df: pd.DataFrame, x_col: str, y_col: str, 
                         future_years: int = 5) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        线性回归预测
        
        Returns:
            (future_x, predicted_y, metrics)
        """
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测未来
        last_year = df[x_col].max()
        future_x = np.arange(last_year + 1, last_year + future_years + 1).reshape(-1, 1)
        predicted_y = model.predict(future_x)
        
        # 计算指标
        y_pred_train = model.predict(X)
        metrics = {
            'r2': r2_score(y, y_pred_train),
            'mae': mean_absolute_error(y, y_pred_train),
            'slope': model.coef_[0],
            'intercept': model.intercept_
        }
        
        self.models['linear'] = model
        return future_x.flatten(), predicted_y, metrics
    
    def polynomial_regression(self, df: pd.DataFrame, x_col: str, y_col: str,
                             degree: int = 2, future_years: int = 5) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        多项式回归预测
        """
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', Ridge(alpha=0.1))
        ])
        model.fit(X, y)
        
        # 预测未来
        last_year = df[x_col].max()
        future_x = np.arange(last_year + 1, last_year + future_years + 1).reshape(-1, 1)
        predicted_y = model.predict(future_x)
        
        # 指标
        y_pred_train = model.predict(X)
        metrics = {
            'r2': r2_score(y, y_pred_train),
            'mae': mean_absolute_error(y, y_pred_train),
            'degree': degree
        }
        
        self.models['polynomial'] = model
        return future_x.flatten(), predicted_y, metrics
    
    def exponential_growth(self, df: pd.DataFrame, x_col: str, y_col: str,
                          future_years: int = 5) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        指数增长模型 (适用于GDP等增长指标)
        y = a * e^(b*x)
        """
        X = df[x_col].values
        y = df[y_col].values
        
        # 定义指数函数
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        try:
            # 归一化年份以避免数值溢出
            x_norm = X - X.min()
            popt, pcov = curve_fit(exp_func, x_norm, y, p0=[y[0], 0.05], maxfev=5000)
            
            # 预测未来
            last_year = X.max()
            future_x = np.arange(last_year + 1, last_year + future_years + 1)
            future_x_norm = future_x - X.min()
            predicted_y = exp_func(future_x_norm, *popt)
            
            # 指标
            y_pred_train = exp_func(x_norm, *popt)
            metrics = {
                'r2': r2_score(y, y_pred_train),
                'mae': mean_absolute_error(y, y_pred_train),
                'params': {'a': popt[0], 'b': popt[1]}
            }
            
            self.models['exponential'] = popt
            return future_x, predicted_y, metrics
            
        except Exception as e:
            # 回退到线性
            return self.linear_regression(df, x_col, y_col, future_years)
    
    def moving_average_forecast(self, series: pd.Series, window: int = 3, 
                                future_periods: int = 5) -> np.ndarray:
        """
        移动平均预测
        """
        ma = series.rolling(window=window).mean().iloc[-1]
        return np.full(future_periods, ma)
    
    def arima_forecast(self, series: pd.Series, order: Tuple = (1,1,1),
                      future_periods: int = 5) -> Tuple[np.ndarray, dict]:
        """
        ARIMA时间序列预测 (简化版，实际需安装statsmodels)
        这里使用简单的自回归替代
        """
        try:
            from statsmodels.tsa.ar_model import AutoReg
            
            model = AutoReg(series, lags=2)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(series), end=len(series)+future_periods-1)
            
            metrics = {'aic': model_fit.aic}
            return predictions.values, metrics
            
        except ImportError:
            # 如果没有statsmodels，使用简单平均增长
            avg_growth = series.pct_change().mean()
            last_val = series.iloc[-1]
            predictions = [last_val * ((1 + avg_growth) ** i) for i in range(1, future_periods+1)]
            return np.array(predictions), {}


class ComprehensivePredictor:
    """综合预测系统"""
    
    def __init__(self):
        self.predictor = TrendPredictor()
        self.predictions = {}
        
    def predict_all_metrics(self, data_dict: Dict[str, pd.DataFrame], 
                           future_years: int = 5) -> Dict:
        """
        对所有指标进行预测
        
        Returns:
            包含预测结果和最佳模型的字典
        """
        results = {}
        
        # GDP预测
        if 'gdp' in data_dict:
            df = data_dict['gdp']
            
            # 尝试三种模型
            models_results = {}
            
            # 1. 线性
            fx, fy, m1 = self.predictor.linear_regression(df, 'year', 'gdp_total', future_years)
            models_results['linear'] = {'x': fx, 'y': fy, 'metrics': m1}
            
            # 2. 多项式
            fx, fy, m2 = self.predictor.polynomial_regression(df, 'year', 'gdp_total', 2, future_years)
            models_results['polynomial'] = {'x': fx, 'y': fy, 'metrics': m2}
            
            # 3. 指数
            fx, fy, m3 = self.predictor.exponential_growth(df, 'year', 'gdp_total', future_years)
            models_results['exponential'] = {'x': fx, 'y': fy, 'metrics': m3}
            
            # 选择R2最高的模型
            best_model = max(models_results.items(), key=lambda x: x[1]['metrics']['r2'])
            
            results['gdp'] = {
                'predictions': models_results,
                'best_model': best_model[0],
                'best_result': best_model[1]
            }
        
        # 人口预测
        if 'population' in data_dict:
            df = data_dict['population']
            fx, fy, m = self.predictor.linear_regression(df, 'year', 'total_population', future_years)
            results['population'] = {'x': fx, 'y': fy, 'metrics': m}
        
        # CPI预测 (使用移动平均，因为波动大)
        if 'cpi' in data_dict:
            df = data_dict['cpi']
            predictions = self.predictor.moving_average_forecast(df['cpi_yoy'], window=3, future_periods=future_years)
            future_x = np.arange(df['year'].max()+1, df['year'].max()+future_years+1)
            results['cpi'] = {'x': future_x, 'y': predictions}
        
        return results
