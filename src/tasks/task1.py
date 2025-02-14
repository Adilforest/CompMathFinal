import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# ===============================
# Шаг 1. Динамический ввод параметров
# ===============================
# Интервал: x ∈ [-n, n]
n = Decimal(input("Введите значение n (интервал будет [-n, n]): "))
a = Decimal(input("Введите коэффициент a: "))
b = Decimal(input("Введите коэффициент b: "))

# Преобразуем значения Decimal в float для вычислений и построения графика
n_val = float(n)
a_val = float(a)
b_val = float(b)

# ===============================
# Шаг 2. Определение функции f(x)
# ===============================
def f(x):
    """
    Вычисляет значение функции: f(x) = x^4 - a*x^2 + b.
    a_val и b_val используются как коэффициенты.
    """
    return x**4 - a_val * x**2 + b_val

# ===============================
# Шаг 3. Создание данных для построения графика
# ===============================
# Генерируем 1000 равномерно распределённых точек в интервале [-n_val, n_val]
x_vals = np.linspace(-n_val, n_val, 1000)
# Вычисляем f(x) для каждой точки
f_vals = np.array([f(x) for x in x_vals])

# ===============================
# Шаг 4. Построение графика функции
# ===============================
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f_vals, label=f'f(x) = x⁴ - {a_val}x² + {b_val}')
plt.axhline(0, color='black', linewidth=0.5)  # ось x
plt.axvline(0, color='black', linewidth=0.5)  # ось y
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'График функции f(x) = x⁴ - {a_val}x² + {b_val}\nна интервале [-{n_val}, {n_val}]')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# Шаг 5. Приблизительный корень (графический метод)
# ===============================
# Ищем смену знака между соседними точками
approx_root_graph = None  # здесь будет найденный графически корень
for i in range(len(x_vals) - 1):
    if f_vals[i] * f_vals[i+1] < 0:
        # При смене знака выбираем середину интервала между двумя точками
        approx_root_graph = (x_vals[i] + x_vals[i+1]) / 2.0
        sign_change_index = i  # сохраняем индекс для численного метода
        break

if approx_root_graph is None:
    print("Смена знака не обнаружена на интервале. Графический метод не смог найти корень.")
else:
    print("Приблизительный корень, найденный графическим методом:", approx_root_graph)

    # ===============================
    # Шаг 6. Метод бисекции для уточнения корня
    # ===============================
    # Используем интервал [x_left, x_right] с обнаруженной сменой знака
    x_left = x_vals[sign_change_index]
    x_right = x_vals[sign_change_index + 1]

    tol = 1e-10         # Задаём допуск
    max_iter = 1000     # Максимальное число итераций
    iterations = 0

    while (x_right - x_left) > tol and iterations < max_iter:
        x_mid = (x_left + x_right) / 2.0  # Находим середину
        f_mid = f(x_mid)
        # Если значение функции близко к 0, останавливаемся
        if abs(f_mid) < tol:
            break
        # Определяем, в какой половине происходит смена знака
        if f(x_left) * f_mid < 0:
            x_right = x_mid
        else:
            x_left = x_mid
        iterations += 1

    approx_root_bis = (x_left + x_right) / 2.0
    print("Корень, найденный методом бисекции:", approx_root_bis)

    # ===============================
    # Шаг 7. Вычисление абсолютной ошибки
    # ===============================
    abs_error = abs(approx_root_graph - approx_root_bis)
    print("Абсолютная ошибка между графическим методом и методом бисекции:", abs_error)
