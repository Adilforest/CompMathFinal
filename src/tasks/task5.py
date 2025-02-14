import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Шаг 1. Ввод данных от пользователя
# ===============================
# Пользователь вводит начальное значение n
n = float(input("Введите начальное значение n для данных (например, 0): "))

# ===============================
# Шаг 2. Формирование точек данных
# ===============================
# Генерируем 4 точки: x = n, n+1, n+2, n+3
x_data = np.array([n, n+1, n+2, n+3])
# По условию (F.E.) y = e^(x) для каждого x, то есть:
y_data = np.exp(x_data)

# ===============================
# Шаг 3. Экспоненциальное приближение
# ===============================
# Модель: y = A * exp(B * x)
# Логарифмируем: ln(y) = ln(A) + B*x
Y = np.log(y_data)

# Находим коэффициенты линейной регрессии: Y = B*x + ln(A)
coeff = np.polyfit(x_data, Y, 1)
B_fit = coeff[0]
lnA_fit = coeff[1]
A_fit = np.exp(lnA_fit)

print("\nНайденные коэффициенты модели:")
print(f"A = {A_fit:.6f}")
print(f"B = {B_fit:.6f}")

# ===============================
# Шаг 4. Вычисление аппроксимированных значений и вывод таблицы
# ===============================
print("\nТаблица сравнения:")
print("{:<10} {:<15} {:<15} {:<15}".format("x", "y (данные)", "y (модель)", "Ошибка"))
for xi, yi in zip(x_data, y_data):
    yi_fit = A_fit * np.exp(B_fit * xi)
    error = yi - yi_fit
    print(f"{xi:<10.4f} {yi:<15.4f} {yi_fit:<15.4f} {error:<15.4e}")

# ===============================
# Шаг 5. Построение графика
# ===============================
# Для гладкой кривой создаём массив x для подграфика
x_fit = np.linspace(x_data[0], x_data[-1], 100)
y_fit = A_fit * np.exp(B_fit * x_fit)

plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, color='red', label='Исходные данные', zorder=5)
plt.plot(x_fit, y_fit, label='Аппроксимирующая модель', color='blue', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Экспоненциальное приближение: y = A * exp(B*x)')
plt.legend()
plt.grid(True)
plt.show()
