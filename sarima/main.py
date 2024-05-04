from data import select_problem_type_data
from evaluate_forecast_mode import evaluate_forecast
from forecast_mode import forecast_next_breakdown
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    # Выбор типа проблемы и получение данных для этого типа
    selected_problem_type, selected_data, all_data = select_problem_type_data()
    
    # Вывод данных для выбранного типа проблемы
    print(f"Выбранный тип проблемы: {selected_problem_type}")
    print(selected_data.head())
    
    # Разделение данных на обучающую и тестовую выборки
    test_size = 0.2
    train_data, test_data = sklearn_train_test_split(all_data, test_size=test_size, shuffle=False)
    
    # Обучение модели SARIMA
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_invertibility=False)
    fitted_model = model.fit(maxiter=200)  # Увеличиваем максимальное количество итераций

    # Сохранение модели
    fitted_model.save("sarima_model.pkl")
    
    # Прогнозирование
    forecast = fitted_model.forecast(steps=len(test_data))
    
    # Оценка качества прогноза
    rmse, mae, mape, wape = evaluate_forecast(test_data, forecast)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    
    forecasted_breakdown = forecast_next_breakdown(selected_data, fitted_model)
    print("Прогноз даты следующей поломки:", forecasted_breakdown)
    
    
if __name__ == "__main__":
    main()