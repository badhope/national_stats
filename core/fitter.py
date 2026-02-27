"""
高级数据拟合模块
支持多种数学函数拟合、曲线拟合和参数优化
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Tuple, Optional
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline, interp1d
import logging
from abc import ABC, abstractmethod

try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False

try:
    import symfit
    SYMFIT_AVAILABLE = True
except ImportError:
    SYMFIT_AVAILABLE = False

from config import Config


class BaseFitter(ABC):
    """拟合器基类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = None
        self.fitted_function = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """执行拟合"""
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """基于拟合结果进行预测"""
        pass
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估拟合效果"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        
        y_pred = self.predict(x)
        
        # 计算各种评估指标
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }


class PolynomialFitter(BaseFitter):
    """多项式拟合器"""
    
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """多项式拟合"""
        try:
            # 执行多项式拟合
            self.coefficients = np.polyfit(x, y, self.degree)
            self.fitted_function = np.poly1d(self.coefficients)
            self.is_fitted = True
            
            # 计算拟合统计信息
            y_pred = self.fitted_function(x)
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'method': f'Polynomial (degree={self.degree})',
                'coefficients': self.coefficients.tolist(),
                'r_squared': r_squared,
                'residual_std': np.std(residuals),
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"多项式拟合失败: {e}")
            return {'method': 'Polynomial', 'success': False, 'error': str(e)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """多项式预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.fitted_function(x)


class ExponentialFitter(BaseFitter):
    """指数函数拟合器"""
    
    def __init__(self, form: str = 'standard'):
        super().__init__()
        self.form = form  # 'standard' or 'modified'
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """指数函数拟合 y = a * exp(b * x) + c"""
        try:
            if self.form == 'standard':
                # 标准形式: y = a * exp(b * x)
                def exp_func(x, a, b):
                    return a * np.exp(b * x)
                
                # 线性化处理初值估计
                valid_idx = y > 0
                if np.sum(valid_idx) < 3:
                    raise ValueError("有效数据点太少")
                
                x_valid = x[valid_idx]
                y_valid = y[valid_idx]
                log_y = np.log(y_valid)
                
                # 线性回归获得初值
                coeffs = np.polyfit(x_valid, log_y, 1)
                p0 = [np.exp(coeffs[1]), coeffs[0]]
                
            else:
                # 修改形式: y = a * exp(b * x) + c
                def exp_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                p0 = [1.0, 0.1, 0.0]
            
            # 非线性最小二乘拟合
            popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
            
            self.params = popt
            self.fitted_function = lambda x_new: exp_func(x_new, *popt)
            self.is_fitted = True
            
            # 计算统计信息
            y_pred = self.fitted_function(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            param_names = ['a', 'b', 'c'] if self.form == 'modified' else ['a', 'b']
            param_dict = dict(zip(param_names, popt.tolist()))
            
            return {
                'method': f'Exponential ({self.form})',
                'parameters': param_dict,
                'r_squared': r_squared,
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"指数拟合失败: {e}")
            return {'method': 'Exponential', 'success': False, 'error': str(e)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """指数函数预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.fitted_function(x)


class LogisticFitter(BaseFitter):
    """逻辑斯蒂函数拟合器"""
    
    def __init__(self):
        super().__init__()
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """逻辑斯蒂函数拟合 y = L / (1 + exp(-k*(x-x0)))"""
        try:
            # 逻辑斯蒂函数
            def logistic_func(x, L, k, x0):
                return L / (1 + np.exp(-k * (x - x0)))
            
            # 参数初值估计
            L_guess = np.max(y) * 1.1  # 上限
            x0_guess = x[np.argmin(np.abs(y - np.median(y)))]  # 中点
            k_guess = 1.0  # 增长率
            
            p0 = [L_guess, k_guess, x0_guess]
            
            # 执行拟合
            popt, pcov = curve_fit(logistic_func, x, y, p0=p0, maxfev=10000)
            
            self.params = popt
            self.fitted_function = lambda x_new: logistic_func(x_new, *popt)
            self.is_fitted = True
            
            # 计算统计信息
            y_pred = self.fitted_function(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            return {
                'method': 'Logistic',
                'parameters': {
                    'L': popt[0],  # 最大值
                    'k': popt[1],  # 增长率
                    'x0': popt[2]   # 中点
                },
                'r_squared': r_squared,
                'carrying_capacity': popt[0],
                'growth_rate': popt[1],
                'inflection_point': popt[2],
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"逻辑斯蒂拟合失败: {e}")
            return {'method': 'Logistic', 'success': False, 'error': str(e)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """逻辑斯蒂函数预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.fitted_function(x)


class SplineFitter(BaseFitter):
    """样条插值拟合器"""
    
    def __init__(self, degree: int = 3, smoothing: float = None):
        super().__init__()
        self.degree = degree
        self.smoothing = smoothing
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """样条插值拟合"""
        try:
            # 创建样条插值
            self.spline = UnivariateSpline(x, y, k=self.degree, s=self.smoothing)
            self.fitted_function = self.spline
            self.is_fitted = True
            
            # 计算拟合统计信息
            y_pred = self.spline(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            return {
                'method': f'Spline (degree={self.degree})',
                'r_squared': r_squared,
                'knots': len(self.spline.get_knots()),
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"样条拟合失败: {e}")
            return {'method': 'Spline', 'success': False, 'error': str(e)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """样条插值预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.spline(x)


class CustomFunctionFitter(BaseFitter):
    """自定义函数拟合器"""
    
    def __init__(self, func: Callable, param_names: List[str], initial_guess: List[float]):
        super().__init__()
        self.func = func
        self.param_names = param_names
        self.initial_guess = initial_guess
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """自定义函数拟合"""
        try:
            # 执行非线性最小二乘拟合
            popt, pcov = curve_fit(self.func, x, y, p0=self.initial_guess, maxfev=10000)
            
            self.params = popt
            self.fitted_function = lambda x_new: self.func(x_new, *popt)
            self.is_fitted = True
            
            # 计算统计信息
            y_pred = self.fitted_function(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            param_dict = dict(zip(self.param_names, popt.tolist()))
            
            return {
                'method': 'Custom Function',
                'parameters': param_dict,
                'r_squared': r_squared,
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"自定义函数拟合失败: {e}")
            return {'method': 'Custom Function', 'success': False, 'error': str(e)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """自定义函数预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.fitted_function(x)


class AdvancedFitter:
    """高级拟合器 - 自动选择最佳拟合方法"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.best_fitter = None
        self.best_method = None
        self.results = {}
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
            methods: List[str] = None) -> Dict[str, Any]:
        """
        自动拟合并比较多种方法
        
        Args:
            x: 自变量数组
            y: 因变量数组
            methods: 要尝试的拟合方法列表
        
        Returns:
            最佳拟合结果
        """
        if methods is None:
            methods = ['polynomial', 'exponential', 'logistic', 'spline']
        
        fitters = {
            'polynomial': [PolynomialFitter(degree=d) for d in [1, 2, 3, 4]],
            'exponential': [ExponentialFitter(form=form) for form in ['standard', 'modified']],
            'logistic': [LogisticFitter()],
            'spline': [SplineFitter(degree=d) for d in [1, 2, 3]]
        }
        
        best_r2 = -np.inf
        best_result = None
        
        # 尝试所有指定方法
        for method in methods:
            if method in fitters:
                for fitter in fitters[method]:
                    try:
                        result = fitter.fit(x, y)
                        if result.get('success', False):
                            self.results[f"{method}_{getattr(fitter, 'degree', '')}"] = result
                            
                            # 评估效果
                            eval_metrics = fitter.evaluate(x, y)
                            r2 = eval_metrics['r2']
                            
                            if r2 > best_r2:
                                best_r2 = r2
                                best_result = result
                                self.best_fitter = fitter
                                self.best_method = f"{method}_{getattr(fitter, 'degree', '')}"
                    
                    except Exception as e:
                        self.logger.warning(f"{method}拟合失败: {e}")
                        continue
        
        if best_result is None:
            raise ValueError("所有拟合方法都失败了")
        
        # 添加综合信息
        best_result['best_method'] = self.best_method
        best_result['all_results'] = self.results
        
        return best_result
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """使用最佳拟合器进行预测"""
        if self.best_fitter is None:
            raise ValueError("尚未进行拟合")
        return self.best_fitter.predict(x)
    
    def get_fitted_function(self) -> Callable:
        """获取拟合函数"""
        if self.best_fitter is None:
            raise ValueError("尚未进行拟合")
        return self.best_fitter.fitted_function
    
    def extrapolate(self, x_new: np.ndarray, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        外推预测（带置信区间）
        
        Args:
            x_new: 新的x值
            confidence_level: 置信水平
        
        Returns:
            包含预测值和置信区间的字典
        """
        if self.best_fitter is None:
            raise ValueError("尚未进行拟合")
        
        # 基本预测
        y_pred = self.best_fitter.predict(x_new)
        
        # 计算残差标准误差
        # 这里简化处理，实际应用中需要更复杂的统计方法
        y_actual = self.best_fitter.predict(x_new[:len(x_new)//2])  # 假设有一部分真实数据
        residual_std = np.std(y_actual - y_pred[:len(y_actual)]) if len(y_actual) > 0 else 1.0
        
        # 计算置信区间
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        margin_error = z_score * residual_std
        
        return {
            'predicted_values': y_pred.tolist(),
            'confidence_lower': (y_pred - margin_error).tolist(),
            'confidence_upper': (y_pred + margin_error).tolist(),
            'residual_std': residual_std
        }


# ==================== 常用拟合函数 ====================

def fit_trend_analysis(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    趋势分析拟合 - 综合多种方法
    """
    fitter = AdvancedFitter()
    result = fitter.fit(x, y)
    
    # 添加趋势分析信息
    trend_info = {
        'trend_direction': 'increasing' if np.mean(np.diff(y)) > 0 else 'decreasing',
        'trend_strength': abs(np.corrcoef(x, y)[0, 1]),
        'volatility': np.std(np.diff(y)),
        'acceleration': np.mean(np.diff(np.diff(y)))  # 二阶差分均值
    }
    
    result.update(trend_info)
    return result


def fit_economic_cycle(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    经济周期拟合 - 识别周期性模式
    """
    from scipy.signal import find_peaks
    
    # 找到峰值和谷值
    peaks, _ = find_peaks(y, height=np.mean(y))
    valleys, _ = find_peaks(-y, height=-np.mean(y))
    
    cycle_info = {
        'peak_dates': x[peaks].tolist(),
        'valley_dates': x[valleys].tolist(),
        'cycle_count': len(peaks),
        'average_cycle_length': np.mean(np.diff(x[peaks])) if len(peaks) > 1 else 0,
        'amplitude': np.mean(y[peaks] - y[valleys]) if len(peaks) > 0 and len(valleys) > 0 else 0
    }
    
    return cycle_info


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * np.exp(0.3 * x) + np.random.normal(0, 0.5, 50)  # 指数增长加噪声
    
    # 使用高级拟合器
    fitter = AdvancedFitter()
    result = fitter.fit(x, y)
    
    print("最佳拟合方法:", result['best_method'])
    print("R²得分:", result['r_squared'])
    print("所有方法结果:", list(result['all_results'].keys()))
    
    # 预测新值
    x_new = np.linspace(10, 15, 10)
    y_pred = fitter.predict(x_new)
    print("外推预测:", y_pred[:5])