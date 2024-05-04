import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Tuple

def evaluate_forecast(test_data: pd.Series, forecast: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Оценка качества прогноза.
    
    Parameters
    ----------
    test_data : pd.Series
        Тестовые данные.
    forecast : np.ndarray
        Прогноз.
    
    Returns
    -------
    Tuple[float, float, float, float]
        Кортеж с метриками качества прогноза (RMSE, MAE, MAPE, WAPE).
    """
    
    mse = mean_squared_error(test_data, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, forecast)
    mape = mean_absolute_percentage_error(test_data, forecast)
    wape = np.sum(np.abs(test_data - forecast)) / np.sum(np.abs(test_data)) * 100
    return rmse, mae, mape, wape
