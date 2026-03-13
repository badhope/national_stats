#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强预测模型模块
宏观经济数据分析平台 - 多种预测算法
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class BasePredictor:
    """预测器基类"""
    
    def __init__(self, name: str = "Base"):
        self.name = name
        self.model = None
        self.history = []
    
    def fit(self, years: List[int], values: List[float]) -> 'BasePredictor':
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, future_years: List[int]) -> List[float]:
        """预测"""
        raise NotImplementedError
    
    def evaluate(self, years: List[int], values: List[float]) -> Dict[str, float]:
        """评估模型"""
        if not self.model:
            return {}
        
        predictions = self.predict(years)
        mse = mean_squared_error(values, predictions) if HAS_SKLEARN else np.mean((np.array(values) - np.array(predictions)) ** 2)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse)
        }


class LinearTrendPredictor(BasePredictor):
    """线性趋势预测"""
    
    def __init__(self):
        super().__init__("线性趋势")
        self.slope = None
        self.intercept = None
    
    def fit(self, years: List[int], values: List[float]) -> 'LinearTrendPredictor':
        if HAS_SKLEARN:
            X = np.array(years).reshape(-1, 1)
            y = np.array(values)
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.slope = self.model.coef_[0]
            self.intercept = self.model.intercept_
        else:
            n = len(years)
            sum_x = sum(years)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(years, values))
            sum_xx = sum(x * x for x in years)
            
            self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            self.intercept = (sum_y - self.slope * sum_x) / n
        
        self.history = list(zip(years, values))
        return self
    
    def predict(self, future_years: List[int]) -> List[float]:
        if HAS_SKLEARN and self.model:
            X = np.array(future_years).reshape(-1, 1)
            return self.model.predict(X).tolist()
        
        return [self.slope * year + self.intercept for year in future_years]


class PolynomialPredictor(BasePredictor):
    """多项式回归预测"""
    
    def __init__(self, degree: int = 2):
        super().__init__(f"多项式({degree}次)")
        self.degree = degree
    
    def fit(self, years: List[int], values: List[float]) -> 'PolynomialPredictor':
        if not HAS_SKLEARN:
            logger.warning("sklearn 未安装，使用线性预测")
            return LinearTrendPredictor().fit(years, values)
        
        X = np.array(years).reshape(-1, 1)
        y = np.array(values)
        
        poly = PolynomialFeatures(degree=self.degree)
        X_poly = poly.fit_transform(X)
        
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_poly, y)
        self.poly = poly
        
        return self
    
    def predict(self, future_years: List[int]) -> List[float]:
        if not HAS_SKLEARN or not self.model:
            return LinearTrendPredictor().fit(*zip(*self.history)).predict(future_years)
        
        X = np.array(future_years).reshape(-1, 1)
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly).tolist()


class ExponentialSmoothingPredictor(BasePredictor):
    """指数平滑预测"""
    
    def __init__(self, alpha: float = 0.3):
        super().__init__("指数平滑")
        self.alpha = alpha
        self.smoothed = None
    
    def fit(self, years: List[int], values: List[float]) -> 'ExponentialSmoothingPredictor':
        self.smoothed = values[0]
        for value in values[1:]:
            self.smoothed = self.alpha * value + (1 - self.alpha) * self.smoothed
        
        self.last_value = values[-1]
        self.trend = (values[-1] - values[0]) / (len(values) - 1)
        
        return self
    
    def predict(self, future_years: List[int]) -> List[float]:
        predictions = []
        current = self.last_value
        
        for _ in future_years:
            current = self.alpha * current + (1 - self.alpha) * (current + self.trend)
            predictions.append(current)
        
        return predictions


class ARIMAPredictor(BasePredictor):
    """简化的ARIMA预测（使用移动平均）"""
    
    def __init__(self, window: int = 3):
        super().__init__("移动平均")
        self.window = window
    
    def fit(self, years: List[int], values: List[float]) -> 'ARIMAPredictor':
        self.values = values
        self.mean = np.mean(values)
        
        if len(values) >= 2:
            self.trend = (values[-1] - values[0]) / (len(values) - 1)
        else:
            self.trend = 0
        
        return self
    
    def predict(self, future_years: List[int]) -> List[float]:
        if not self.values:
            return []
        
        predictions = []
        base_value = self.values[-1]
        
        for i, _ in enumerate(future_years):
            predicted = base_value + self.trend * (i + 1)
            predictions.append(predicted)
        
        return predictions


class GrowthRatePredictor(BasePredictor):
    """增长率预测"""
    
    def __init__(self):
        super().__init__("增长率")
    
    def fit(self, years: List[int], values: List[float]) -> 'GrowthRatePredictor':
        if len(values) < 2:
            self.avg_growth_rate = 0
            return self
        
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                rate = (values[i] - values[i-1]) / values[i-1]
                growth_rates.append(rate)
        
        self.avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
        self.last_value = values[-1]
        
        return self
    
    def predict(self, future_years: List[int]) -> List[float]:
        predictions = []
        current = self.last_value
        
        for _ in future_years:
            current = current * (1 + self.avg_growth_rate)
            predictions.append(current)
        
        return predictions


class EnsemblePredictor:
    """集成预测器 - 组合多种预测方法"""
    
    def __init__(self):
        self.predictors = [
            LinearTrendPredictor(),
            PolynomialPredictor(degree=2),
            ExponentialSmoothingPredictor(alpha=0.3),
            GrowthRatePredictor()
        ]
        self.weights = None
    
    def fit(self, years: List[int], values: List[float]) -> 'EnsemblePredictor':
        for predictor in self.predictors:
            predictor.fit(years, values)
        
        predictions = np.array([p.predict(years) for p in self.predictors])
        errors = np.abs(predictions - np.array(values))
        
        weights = 1 / (errors.mean(axis=1) + 1e-10)
        self.weights = weights / weights.sum()
        
        return self
    
    def predict(self, future_years: List[int]) -> List[float]:
        predictions = np.array([p.predict(future_years) for p in self.predictors])
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_predictions.tolist()
    
    def get_forecast_range(self, future_years: List[int]) -> Tuple[List[float], List[float]]:
        """获取预测区间"""
        all_predictions = np.array([p.predict(future_years) for p in self.predictors])
        
        lower = np.min(all_predictions, axis=0)
        upper = np.max(all_predictions, axis=0)
        
        return lower.tolist(), upper.tolist()


class EconomicPredictor:
    """宏观经济预测主类"""
    
    def __init__(self, method: str = 'ensemble'):
        self.method = method
        
        if method == 'linear':
            self.predictor = LinearTrendPredictor()
        elif method == 'polynomial':
            self.predictor = PolynomialPredictor(degree=2)
        elif method == 'exponential':
            self.predictor = ExponentialSmoothingPredictor()
        elif method == 'growth':
            self.predictor = GrowthRatePredictor()
        elif method == 'ensemble':
            self.predictor = EnsemblePredictor()
        else:
            self.predictor = LinearTrendPredictor()
    
    def predict(self, years: List[int], values: List[float], 
                future_years: int = 5) -> Dict[str, Any]:
        """执行预测
        
        Args:
            years: 历史年份
            values: 历史数值
            future_years: 预测未来年数
        
        Returns:
            预测结果字典
        """
        future_year_list = list(range(max(years) + 1, max(years) + future_years + 1))
        
        self.predictor.fit(years, values)
        
        predictions = self.predictor.predict(future_year_list)
        
        result = {
            'method': self.method,
            'history': {
                'years': years,
                'values': values
            },
            'forecast': {
                'years': future_year_list,
                'values': [round(v, 2) for v in predictions]
            },
            'summary': {
                'base_value': values[-1],
                'predicted_value': predictions[-1],
                'total_change': predictions[-1] - values[-1],
                'change_rate': (predictions[-1] - values[-1]) / values[-1] * 100 if values[-1] != 0 else 0
            }
        }
        
        if isinstance(self.predictor, EnsemblePredictor):
            lower, upper = self.predictor.get_forecast_range(future_year_list)
            result['forecast']['lower_bound'] = [round(v, 2) for v in lower]
            result['forecast']['upper_bound'] = [round(v, 2) for v in upper]
        
        return result
    
    def compare_methods(self, years: List[int], values: List[float],
                       future_years: int = 5) -> Dict[str, Any]:
        """比较多种预测方法"""
        methods = ['linear', 'polynomial', 'exponential', 'growth', 'ensemble']
        results = {}
        
        for method in methods:
            try:
                predictor = EconomicPredictor(method)
                result = predictor.predict(years, values, future_years)
                results[method] = result['summary']
            except Exception as e:
                logger.error(f"方法 {method} 预测失败: {e}")
        
        return results


def predict_gdp(years: List[int], values: List[float], 
                future_years: int = 5, method: str = 'ensemble') -> Dict[str, Any]:
    """GDP预测便捷函数"""
    predictor = EconomicPredictor(method)
    return predictor.predict(years, values, future_years)


def predict_cpi(years: List[int], values: List[float],
                future_years: int = 5, method: str = 'ensemble') -> Dict[str, Any]:
    """CPI预测便捷函数"""
    predictor = EconomicPredictor(method)
    return predictor.predict(years, values, future_years)


if __name__ == '__main__':
    test_years = [2020, 2021, 2022, 2023, 2024]
    test_values = [101.6, 110.4, 121.0, 121.0, 126.1]
    
    print("=" * 50)
    print("宏观经济预测模型测试")
    print("=" * 50)
    
    predictor = EconomicPredictor(method='ensemble')
    result = predictor.predict(test_years, test_values, future_years=3)
    
    print(f"\n预测方法: {result['method']}")
    print(f"\n历史数据:")
    for year, value in zip(result['history']['years'], result['history']['values']):
        print(f"  {year}: {value}")
    
    print(f"\n预测结果:")
    for year, value in zip(result['forecast']['years'], result['forecast']['values']):
        print(f"  {year}: {value}")
    
    print(f"\n预测摘要:")
    print(f"  基准值: {result['summary']['base_value']}")
    print(f"  预测值: {result['summary']['predicted_value']}")
    print(f"  变化量: {result['summary']['total_change']:.2f}")
    print(f"  变化率: {result['summary']['change_rate']:.2f}%")
    
    if 'lower_bound' in result['forecast']:
        print(f"\n预测区间:")
        for year, lower, upper in zip(result['forecast']['years'],
                                       result['forecast']['lower_bound'],
                                       result['forecast']['upper_bound']):
            print(f"  {year}: [{lower:.2f}, {upper:.2f}]")
    
    print("\n" + "=" * 50)
    print("方法比较:")
    print("=" * 50)
    
    comparison = predictor.compare_methods(test_years, test_values, future_years=3)
    for method, summary in comparison.items():
        print(f"\n{method}:")
        print(f"  预测值: {summary['predicted_value']:.2f}")
        print(f"  变化率: {summary['change_rate']:.2f}%")
