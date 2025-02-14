import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ===============================
# Шаг 1. Ввод исходных данных пользователем
# ===============================
# Введите значения x через пробел (например: 0 0.5 1.0 1.5)
x_input = input("Введите значения x через пробел (например, 0 0.5 1.0 1.5): ")
x_data = np.array([float(val) for val in x_input.split()])

# Введите значения y через пробел (например: 0 0.25 0.75 2.25)
y_input = input("Введите значения y через пробел (например, 0 0.25 0.75 2.25): ")
y_data = np.array([float(val) for val in y_input.split()])

# Проверка: число введённых значений должно совпадать
if len(x_data) != len(y_data):
    print("Ошибка: количество значений x и y должно совпадать!")
    exit()

# ===============================
# Шаг 2. Создание интерполятора кубическим сплайном
# ===============================
# Используем натуральные (второй производной равной 0 на концах) условия
cs = CubicSpline(x_data, y_data, bc_type='natural')

# ===============================
# Шаг 3. Формирование данных для графика и таблицы
# ===============================
# Генерируем плотную сетку для построения кривой интерполяции
x_interp = np.linspace(np.min(x_data), np.max(x_data), 200)
y_interp = cs(x_interp)

# Для таблицы выберем 10 равномерно распределённых точек
x_table = np.linspace(np.min(x_data), np.max(x_data), 10)
y_table = cs(x_table)

# Вывод таблицы интерполированных значений
print("\nТаблица интерполированных значений:")
print("{:<10} {:<15}".format("x", "Spline(y)"))
for xi, yi in zip(x_table, y_table):
    print("{:<10.4f} {:<15.4f}".format(xi, yi))

# ===============================
# Шаг 4. Построение графика
# ===============================
plt.figure(figsize=(8, 6))
plt.plot(x_interp, y_interp, label='Кубический сплайн', color='blue', linewidth=2)
plt.scatter(x_data, y_data, color='red', label='Исходные данные', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Кубическая сплайн-интерполяция')
plt.legend()
plt.grid(True)
plt.show()
