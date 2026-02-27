"""
高级预测器模块
支持多种时间序列预测方法，包括传统统计方法和机器学习方法
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# 传统统计方法
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

# 机器学习方法
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Facebook Prophet (如果可用)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# XGBoost (如果可用)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import Config
from models.time_series import MacroTimeSeries


class BasePredictor(ABC):
    """预测器基类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> Dict[str, Any]:
        """进行预测"""
        pass
    
    def evaluate(self, ts: MacroTimeSeries, test_periods: int = 12) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            ts: 时间序列数据
            test_periods: 测试期数
            
        Returns:
            评估指标字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 分割数据
        train_data = ts.data[:-test_periods]
        test_data = ts.data[-test_periods:]
        
        if len(train_data) < 10:
            raise ValueError("训练数据不足")
        
        # 重新训练模型（使用训练集）
        temp_ts = MacroTimeSeries(
            data=train_data.copy(),
            meta=ts.meta
        )
        self.fit(temp_ts)
        
        # 预测测试集
        predictions = self.predict(len(test_data))
        pred_values = predictions['forecast']
        
        # 计算评估指标
        actual_values = test_data['value'].values
        
        metrics = {
            'mse': mean_squared_error(actual_values, pred_values),
            'rmse': np.sqrt(mean_squared_error(actual_values, pred_values)),
            'mae': mean_absolute_error(actual_values, pred_values),
            'mape': np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100,
            'r2': r2_score(actual_values, pred_values)
        }
        
        return metrics


class ARIMAPredictor(BasePredictor):
    """ARIMA预测器"""
    
    def __init__(self, order: Tuple[int, int, int] = None):
        super().__init__()
        self.order = order or (1, 1, 1)  # 默认参数
    
    def fit(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """训练ARIMA模型"""
        try:
            # 准备数据
            y = ts.data['value'].values
            
            # 自动选择最优参数（如果未指定）
            if self.order == (1, 1, 1):
                self.order = self._auto_arima_order(y)
            
            # 训练模型
            self.model = ARIMA(y, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # 返回训练信息
            return {
                'method': 'ARIMA',
                'order': self.order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"ARIMA训练失败: {e}")
            return {'method': 'ARIMA', 'success': False, 'error': str(e)}
    
    def predict(self, periods: int) -> Dict[str, Any]:
        """ARIMA预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        try:
            # 进行预测
            forecast_result = self.fitted_model.forecast(steps=periods)
            forecast_values = forecast_result.tolist()
            
            # 计算置信区间
            forecast_ci = self.fitted_model.get_forecast(steps=periods)
            ci_lower = forecast_ci.conf_int().iloc[:, 0].tolist()
            ci_upper = forecast_ci.conf_int().iloc[:, 1].tolist()
            
            return {
                'method': 'ARIMA',
                'forecast': forecast_values,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'r_squared': self.fitted_model.rsquared if hasattr(self.fitted_model, 'rsquared') else None
            }
        
        except Exception as e:
            self.logger.error(f"ARIMA预测失败: {e}")
            return {'method': 'ARIMA', 'success': False, 'error': str(e)}
    
    def _auto_arima_order(self, y: np.ndarray) -> Tuple[int, int, int]:
        """自动选择ARIMA参数"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # 限制搜索范围以提高效率
        max_p = min(Config.model.arima_max_p, 3)
        max_d = min(Config.model.arima_max_d, 2)
        max_q = min(Config.model.arima_max_q, 3)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order


class ProphetPredictor(BasePredictor):
    """Facebook Prophet预测器"""
    
    def __init__(self):
        super().__init__()
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet库未安装，请运行: pip install prophet")
    
    def fit(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """训练Prophet模型"""
        try:
            # 准备数据格式
            df = ts.data.copy()
            df = df.rename(columns={'date': 'ds', 'value': 'y'})
            
            # 创建并训练模型
            self.model = Prophet(
                seasonality_mode=Config.model.prophet_seasonality_mode,
                yearly_seasonality=Config.model.prophet_yearly_seasonality,
                weekly_seasonality=Config.model.prophet_weekly_seasonality,
                daily_seasonality=Config.model.prophet_daily_seasonality
            )
            
            self.model.fit(df)
            self.is_fitted = True
            
            return {
                'method': 'Prophet',
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"Prophet训练失败: {e}")
            return {'method': 'Prophet', 'success': False, 'error': str(e)}
    
    def predict(self, periods: int) -> Dict[str, Any]:
        """Prophet预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        try:
            # 创建未来日期框架
            future = self.model.make_future_dataframe(periods=periods, freq='MS')
            forecast = self.model.predict(future)
            
            # 提取预测结果
            last_date = self.model.history['ds'].max()
            future_forecast = forecast[forecast['ds'] > last_date]
            
            forecast_values = future_forecast['yhat'].tolist()
            ci_lower = future_forecast['yhat_lower'].tolist()
            ci_upper = future_forecast['yhat_upper'].tolist()
            
            return {
                'method': 'Prophet',
                'forecast': forecast_values,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper
            }
        
        except Exception as e:
            self.logger.error(f"Prophet预测失败: {e}")
            return {'method': 'Prophet', 'success': False, 'error': str(e)}


class XGBoostPredictor(BasePredictor):
    """XGBoost预测器"""
    
    def __init__(self):
        super().__init__()
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost库未安装，请运行: pip install xgboost")
        self.scaler = StandardScaler()
    
    def fit(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """训练XGBoost模型"""
        try:
            # 准备特征
            X, y = self._prepare_features(ts.data)
            
            # 标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 创建并训练模型
            self.model = xgb.XGBRegressor(
                n_estimators=Config.model.xgb_n_estimators,
                max_depth=Config.model.xgb_max_depth,
                learning_rate=Config.model.xgb_learning_rate,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            # 计算训练集R²
            train_pred = self.model.predict(X_scaled)
            r2 = r2_score(y, train_pred)
            
            return {
                'method': 'XGBoost',
                'r_squared': r2,
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"XGBoost训练失败: {e}")
            return {'method': 'XGBoost', 'success': False, 'error': str(e)}
    
    def predict(self, periods: int) -> Dict[str, Any]:
        """XGBoost预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        try:
            # 获取最后几个数据点作为预测起点
            last_data = self.last_window.copy()
            predictions = []
            
            for i in range(periods):
                # 准备特征
                X = self._create_features_for_prediction(last_data)
                X_scaled = self.scaler.transform(X.reshape(1, -1))
                
                # 预测
                pred = self.model.predict(X_scaled)[0]
                predictions.append(pred)
                
                # 更新窗口数据
                last_data = np.roll(last_data, -1)
                last_data[-1] = pred
            
            return {
                'method': 'XGBoost',
                'forecast': predictions,
                'r_squared': None  # 预测时无法计算R²
            }
        
        except Exception as e:
            self.logger.error(f"XGBoost预测失败: {e}")
            return {'method': 'XGBoost', 'success': False, 'error': str(e)}
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练特征"""
        values = df['value'].values
        dates = pd.to_datetime(df['date'])
        
        features = []
        targets = []
        
        # 使用过去12个月的数据作为特征
        window_size = 12
        
        for i in range(window_size, len(values)):
            # 历史值特征
            hist_values = values[i-window_size:i]
            
            # 时间特征
            current_date = dates.iloc[i]
            time_features = [
                current_date.month,
                current_date.quarter,
                current_date.year
            ]
            
            # 组合特征
            feature_vector = np.concatenate([hist_values, time_features])
            features.append(feature_vector)
            targets.append(values[i])
        
        self.last_window = values[-window_size:]  # 保存最后窗口用于预测
        return np.array(features), np.array(targets)
    
    def _create_features_for_prediction(self, window_data: np.ndarray) -> np.ndarray:
        """为预测创建特征向量"""
        # 使用当前时间
        current_date = datetime.now()
        time_features = [
            current_date.month,
            current_date.quarter,
            current_date.year
        ]
        
        return np.concatenate([window_data, time_features])


class EnsemblePredictor(BasePredictor):
    """集成预测器 - 组合多种预测方法"""
    
    def __init__(self, methods: List[str] = None):
        super().__init__()
        self.methods = methods or ['arima', 'prophet', 'xgboost']
        self.predictors = {}
        self.weights = {}  # 各方法权重
    
    def fit(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """训练所有预测器"""
        results = {}
        
        # 初始化各预测器
        if 'arima' in self.methods:
            self.predictors['arima'] = ARIMAPredictor()
        
        if 'prophet' in self.methods and PROPHET_AVAILABLE:
            self.predictors['prophet'] = ProphetPredictor()
        
        if 'xgboost' in self.methods and XGBOOST_AVAILABLE:
            self.predictors['xgboost'] = XGBoostPredictor()
        
        # 训练各预测器并计算权重
        weights_sum = 0
        for name, predictor in self.predictors.items():
            try:
                result = predictor.fit(ts)
                if result.get('success', False):
                    # 基于AIC或其他指标设置权重
                    if name == 'arima' and 'aic' in result:
                        weight = 1 / (1 + abs(result['aic']))  # AIC越小权重越大
                    else:
                        weight = 1.0
                    
                    self.weights[name] = weight
                    weights_sum += weight
                    results[name] = result
                else:
                    self.logger.warning(f"{name}训练失败: {result.get('error', 'Unknown error')}")
            except Exception as e:
                self.logger.error(f"{name}训练异常: {e}")
        
        # 归一化权重
        if weights_sum > 0:
            for name in self.weights:
                self.weights[name] /= weights_sum
        
        self.is_fitted = len(self.predictors) > 0
        
        return {
            'method': 'Ensemble',
            'individual_results': results,
            'weights': self.weights,
            'success': self.is_fitted
        }
    
    def predict(self, periods: int) -> Dict[str, Any]:
        """集成预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 收集各预测器的结果
        individual_predictions = {}
        for name, predictor in self.predictors.items():
            try:
                pred_result = predictor.predict(periods)
                if pred_result.get('success', True):  # Prophet没有success字段
                    individual_predictions[name] = pred_result['forecast']
            except Exception as e:
                self.logger.error(f"{name}预测失败: {e}")
        
        if not individual_predictions:
            raise ValueError("所有预测器都失败了")
        
        # 加权平均集成
        ensemble_forecast = np.zeros(periods)
        total_weight = 0
        
        for name, predictions in individual_predictions.items():
            weight = self.weights.get(name, 1.0 / len(individual_predictions))
            ensemble_forecast += np.array(predictions) * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_forecast /= total_weight
        
        return {
            'method': 'Ensemble',
            'forecast': ensemble_forecast.tolist(),
            'individual_predictions': individual_predictions,
            'weights': self.weights
        }


class Predictor:
    """主预测器类 - 提供统一接口"""
    
    def __init__(self, method: str = 'auto'):
        """
        初始化预测器
        
        Args:
            method: 预测方法 ('arima', 'prophet', 'xgboost', 'ensemble', 'auto')
        """
        self.method = method
        self.predictor = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forecast(self, ts: MacroTimeSeries, periods: int = 12, 
                 method: str = None) -> Dict[str, Any]:
        """
        执行预测
        
        Args:
            ts: 时间序列数据
            periods: 预测期数
            method: 预测方法（可覆盖初始化时的方法）
        
        Returns:
            预测结果字典
        """
        method = method or self.method
        
        # 选择预测器
        if method == 'auto':
            # 自动选择最适合的方法
            method = self._select_best_method(ts)
        
        self._initialize_predictor(method)
        
        if self.predictor is None:
            raise ValueError(f"不支持的预测方法: {method}")
        
        # 训练模型
        train_result = self.predictor.fit(ts)
        if not train_result.get('success', True):
            raise ValueError(f"模型训练失败: {train_result.get('error', 'Unknown error')}")
        
        # 执行预测
        prediction_result = self.predictor.predict(periods)
        
        # 生成预测日期
        last_date = pd.to_datetime(ts.data['date'].iloc[-1])
        freq_map = {
            'monthly': 'MS',
            'quarterly': 'QS',
            'yearly': 'YS'
        }
        freq = freq_map.get(ts.meta.indicator.frequency, 'MS')
        
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=periods,
            freq=freq
        )
        
        # 组合完整结果
        result = {
            'method': prediction_result['method'],
            'forecast': prediction_result['forecast'],
            'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'training_info': train_result,
            'periods': periods
        }
        
        # 添加置信区间（如果可用）
        if 'confidence_interval_lower' in prediction_result:
            result['confidence_interval_lower'] = prediction_result['confidence_interval_lower']
            result['confidence_interval_upper'] = prediction_result['confidence_interval_upper']
        
        # 添加评估指标
        try:
            evaluation = self.predictor.evaluate(ts, min(12, len(ts)//4))
            result['evaluation'] = evaluation
        except Exception as e:
            self.logger.warning(f"模型评估失败: {e}")
        
        return result
    
    def _select_best_method(self, ts: MacroTimeSeries) -> str:
        """自动选择最佳预测方法"""
        data_length = len(ts)
        
        # 基于数据长度选择
        if data_length < 20:
            return 'arima'  # 数据较少时使用ARIMA
        elif data_length < 100:
            return 'prophet'  # 中等数据量使用Prophet
        else:
            return 'ensemble'  # 大数据量使用集成方法
    
    def _initialize_predictor(self, method: str):
        """初始化预测器实例"""
        method_map = {
            'arima': ARIMAPredictor,
            'prophet': ProphetPredictor,
            'xgboost': XGBoostPredictor,
            'ensemble': EnsemblePredictor
        }
        
        if method in method_map:
            try:
                self.predictor = method_map[method]()
            except ImportError as e:
                self.logger.warning(f"{method}不可用: {e}")
                # 尝试其他方法
                fallback_methods = ['arima', 'prophet', 'xgboost']
                for fallback in fallback_methods:
                    if fallback != method and fallback in method_map:
                        try:
                            self.predictor = method_map[fallback]()
                            self.logger.info(f"使用备用方法: {fallback}")
                            break
                        except ImportError:
                            continue
        else:
            raise ValueError(f"不支持的预测方法: {method}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例使用
    import numpy as np
    
    # 创建示例数据
    dates = pd.date_range('2020-01-01', periods=60, freq='MS')
    values = 100 + np.cumsum(np.random.randn(60) * 0.5)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    from models.time_series import TimeSeriesMeta
    from config import IndicatorDefinition, IndicatorCategory, IndicatorFrequency
    
    indicator_def = IndicatorDefinition(
        code='sample_gdp',
        name='样本GDP',
        category=IndicatorCategory.PRODUCTION,
        frequency=IndicatorFrequency.MONTHLY,
        unit='亿元',
        source='sample'
    )
    
    meta = TimeSeriesMeta(
        indicator=indicator_def,
        source='sample',
        last_updated=datetime.now()
    )
    
    ts = MacroTimeSeries(data=sample_data, meta=meta)
    
    # 使用预测器
    predictor = Predictor(method='auto')
    result = predictor.forecast(ts, periods=12)
    
    print("预测方法:", result['method'])
    print("预测值:", result['forecast'][:5])  # 显示前5个预测值
    if 'evaluation' in result:
        print("R²得分:", result['evaluation'].get('r2', 'N/A'))