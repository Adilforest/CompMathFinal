from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Шаг 1. Ввод значения x от пользователя
# ===============================
x_input = Decimal(input("Введите значение x для вычисления y (например, 0.2): "))
x_val = float(x_input)

# ===============================
# Шаг 2. Определение функций-приближений (Метод Пикара)
# ===============================
def y0(x):
    # Начальное приближение: постоянная функция y0(x)=1
    return 1.0

def y1(x):
    # Первое приближение: y1(x)=1 + x + x^2/2
    return 1.0 + x + x**2 / 2

def y2(x):
    # Второе приближение: y2(x)=1 + x + x^2 + x^3/6
    return 1.0 + x + x**2 + x**3 / 6

def y3(x):
    # Третье приближение: y3(x)=1 + x + x^2 + x^3/3 + x^4/24
    return 1.0 + x + x**2 + x**3 / 3 + x**4 / 24

def y4(x):
    # Четвёртое приближение: y4(x)=1 + x + x^2 + x^3/3 + x^4/12 + x^5/120
    return 1.0 + x + x**2 + x**3 / 3 + x**4 / 12 + x**5 / 120

# ===============================
# Шаг 3. Вычисление приближённых значений в точке x_val
# ===============================
y0_val = y0(x_val)
y1_val = y1(x_val)
y2_val = y2(x_val)
y3_val = y3(x_val)
y4_val = y4(x_val)

# ===============================
# Шаг 4. Вывод таблицы с приближениями
# ===============================
print("\nТаблица приближённых значений по методу Пикара:")
print("{:<10} {:<20}".format("Итерация", "y(x)"))
print("{:<10} {:<20.12f}".format("y0", y0_val))
print("{:<10} {:<20.12f}".format("y1", y1_val))
print("{:<10} {:<20.12f}".format("y2", y2_val))
print("{:<10} {:<20.12f}".format("y3", y3_val))
print("{:<10} {:<20.12f}".format("y4", y4_val))

print("\nЗначение y в x = {:.4f} по 4-й аппроксимации: {:.12f}".format(x_val, y4_val))

# ===============================
# Шаг 5. Построение графиков приближений
# ===============================
# Построим графики y0, y1, y2, y3, y4 на интервале [0, 0.5]
x_plot = np.linspace(0, 0.5, 200)
y0_plot = np.array([y0(x) for x in x_plot])
y1_plot = np.array([y1(x) for x in x_plot])
y2_plot = np.array([y2(x) for x in x_plot])
y3_plot = np.array([y3(x) for x in x_plot])
y4_plot = np.array([y4(x) for x in x_plot])

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y0_plot, label="y0(x)=1", linestyle="--")
plt.plot(x_plot, y1_plot, label="y1(x)=1+x+x²/2", linestyle="-")
plt.plot(x_plot, y2_plot, label="y2(x)=1+x+x²+x³/6", linestyle="-.")
plt.plot(x_plot, y3_plot, label="y3(x)=1+x+x²+x³/3+x⁴/24", linestyle=":")
plt.plot(x_plot, y4_plot, label="y4(x)=1+x+x²+x³/3+x⁴/12+x⁵/120", linewidth=2)

# Отмечаем значение в точке x_val для каждого приближения
plt.scatter([x_val]*5, [y0_val, y1_val, y2_val, y3_val, y4_val], color="red", zorder=5)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Метод Пикара: последовательные приближения решения дифференциального уравнения")
plt.legend()
plt.grid(True)
plt.show()
