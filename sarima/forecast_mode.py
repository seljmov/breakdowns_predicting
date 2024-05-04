import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_next_breakdown(data: pd.DataFrame, trained_model: SARIMAX) -> pd.Timestamp:
    """
    Прогнозирование даты следующей поломки на основе данных и обученной модели SARIMA.
    """
    # Получение последнего доступного временного значения в данных
    last_timestamp = data['timestamp_numeric'].max()

    # Прогнозирование на один шаг вперед
    forecast = trained_model.forecast(steps=1)

    # Получение даты следующей поломки
    next_breakdown_timestamp = last_timestamp + forecast.iloc[0]

    return pd.Timestamp(next_breakdown_timestamp, unit='s')  # Преобразуем обратно в Timestamp
