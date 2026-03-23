import pandas as pd
import joblib

model = joblib.load('taxi_fare_model.pkl')

def predict_fare(distance, duration_hours, base_fare):

    duration_seconds = duration_hours * 3600
    input_data = pd.DataFrame([[distance, duration_seconds, base_fare]],
                              columns=['distance_traveled', 'trip_duration', 'fare'])
    return model.predict(input_data)[0]

while True:
    while True:
        try:
            distance = float(input("Расстояние (км): "))
            break
        except ValueError:
            print("Пожалуйста, введите число!")

    while True:
        try:
            duration = float(input("Время (часы): "))
            break
        except ValueError:
            print("Пожалуйста, введите число!")

    while True:
        try:
            base_fare = float(input("Базовая стоимость (сом): "))
            break
        except ValueError:
            print("Пожалуйста, введите число!")

    result = predict_fare(distance, duration, base_fare)
    print(f"Цена: {result:.2f} сом")

    again = input("Хотите рассчитать ещё раз? (да/нет): ").strip().lower()
    if again not in ('да', 'y', 'yes'):
        print("Программа завершена.")
        break