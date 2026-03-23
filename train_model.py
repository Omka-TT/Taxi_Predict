import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

features = ['distance_traveled', 'trip_duration', 'fare']
target = 'total_fare'

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'taxi_fare_model.pkl')

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred) * 100
mse = mean_squared_error(y_test, y_pred)

total_rows = len(train_data) + len(test_data)
features_count = len(features)

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {r2:.2f}%\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"Total Rows: {total_rows}\n")
    f.write(f"Features: {features_count}\n")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.scatter(y_test, y_pred, c='blue')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title(f'Сравнение фактических и предсказанных цен на такси\n'
          f'Точность модели: {r2:.1f}%')

plt.savefig("static/graph.png")




