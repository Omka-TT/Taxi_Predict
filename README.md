Taxi Price Predictor

Веб-приложение для предсказания стоимости поездки на такси с использованием модели машинного обучения (Linear Regression).

Описание

Этот проект позволяет пользователю ввести параметры поездки и получить предсказанную стоимость.

Модель обучена на данных из train.csv и использует следующие признаки:

distance_traveled — пройденное расстояние
trip_duration — длительность поездки
fare — базовая стоимость

Целевая переменная:

total_fare — итоговая стоимость поездки
- Установка и запуск
1. Клонировать проект
git clone <your-repo-url>
cd project
2. Создать виртуальное окружение
python -m venv .venv
3. Активировать окружение

Windows:

.venv\Scripts\activate

Mac/Linux:

source .venv/bin/activate
4. Установить зависимости
pip install flask numpy pandas scikit-learn joblib matplotlib
5. Запустить приложение
python app.py
6. Открыть в браузере
http://127.0.0.1:5000
