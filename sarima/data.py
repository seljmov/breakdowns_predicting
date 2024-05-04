import pandas as pd
import click

# Функция для загрузки данных и вывода списка доступных типов проблем
def get_data() -> tuple:
    # Загрузка данных из CSV-файла
    data = pd.read_csv("data.csv", parse_dates=['createdTimestamp'], dayfirst=True)
    
    # Разделение строк по запятой в столбце problem_type
    data['problem_type'] = data['problem_type'].str.split(',')
    
    # Создание нового DataFrame с каждым типом проблемы в отдельной строке
    data = data.explode('problem_type')
    
    # Удаление столбца 'address'
    data.drop(columns=['address'], inplace=True)
    
    # Получение уникальных типов проблем
    problem_types = data['problem_type'].unique()
    
    # Вывод списка доступных типов проблем
    click.echo("Выберите тип проблемы:")
    for idx, problem_type in enumerate(problem_types, start=1):
        click.echo(f"{idx} - {problem_type}")
    
    return data, problem_types

# Функция для обработки выбора типа проблемы
def select_problem_type_data() -> tuple:
    """
    Функция для выбора типа проблемы и получения данных для этого типа.
    """
    # Загрузка данных и получение списка доступных типов проблем
    data, problem_types = get_data()
    
    # Запрос выбора типа проблемы
    while True:
        problem_type_index = input("Введите номер типа проблемы: ")
        if problem_type_index.isdigit():
            problem_type_index = int(problem_type_index)
            if 1 <= problem_type_index <= len(problem_types):
                break
        print("Некорректный номер типа проблемы. Пожалуйста, выберите существующий номер.")
    
    # Выбор типа проблемы
    selected_problem_type = problem_types[problem_type_index - 1]
    
    # Получение данных для выбранного типа проблемы
    problem_type_data = data[data['problem_type'] == selected_problem_type].copy()
    
    # Преобразование столбца createdTimestamp в тип datetime
    problem_type_data['createdTimestamp'] = pd.to_datetime(problem_type_data['createdTimestamp'])

    # Преобразование столбца createdTimestamp в числовые значения (количество секунд с начала эпохи)
    problem_type_data['timestamp_numeric'] = problem_type_data['createdTimestamp'].astype(int) / 10**9  # деление на 10^9 для перевода наносекунд в секунды

    # Предположим, что столбец createdTimestamp содержит даты поломок
    selected_data = problem_type_data[['timestamp_numeric']]
    
    return selected_problem_type, selected_data, data
