import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def f(x, a, b, c):
    """
    Вычисляет значение функции f(x) = a*x^4 - b*x^2 + c.
    """
    return a * x**4 - b * x**2 + c

def plot_function(a, b, c, n):
    """
    Строит график функции f(x)= a*x^4 - b*x^2 + c на интервале [-n, n].
    Возвращает массив узловых точек x и соответствующих значений f(x).
    """
    x_vals = np.linspace(-n, n, 1000)
    f_vals = f(x_vals, a, b, c)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, f_vals, label=f'f(x) = {a}x⁴ - {b}x² + {c}', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)  # ось x
    plt.axvline(0, color='black', linewidth=0.5)  # ось y
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'График функции на интервале [-{n}, {n}]')
    plt.legend()
    plt.grid(True)
    plt.show()

    return x_vals, f_vals

def find_graphical_root(x_vals, f_vals):
    """
    Находит приближённый корень методом графического поиска.
    Ищется первая смена знака между соседними точками,
    и корень приближённо определяется как среднее значение этих x.
    Возвращает кортеж (корень, индекс начала интервала).
    """
    for i in range(len(x_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            approx_root = (x_vals[i] + x_vals[i+1]) / 2.0
            return approx_root, i
    return None, None

def bisection_method(f, x_left, x_right, a, b, c, tol=1e-10, max_iter=1000):
    """
    Находит корень функции f(x) = a*x^4 - b*x^2 + c на отрезке [x_left, x_right]
    методом бисекции.

    Возвращает (приближённый корень, число итераций).
    """
    iterations = 0
    left = x_left
    right = x_right

    while (right - left) > tol and iterations < max_iter:
        mid = (left + right) / 2.0
        f_mid = f(mid, a, b, c)

        if abs(f_mid) < tol:
            return mid, iterations

        if f(left, a, b, c) * f_mid < 0:
            right = mid
        else:
            left = mid

        iterations += 1

    return (left + right) / 2.0, iterations

def main():
    # Ввод коэффициентов и интервала
    n_input = Decimal(input("Введите значение n (интервал будет [-n, n]): "))
    a_input = Decimal(input("Введите коэффициент a (для x^4): "))
    b_input = Decimal(input("Введите коэффициент b (для x^2): "))
    c_input = Decimal(input("Введите коэффициент c (свободный член): "))

    # Преобразование в float
    n_val = float(n_input)
    a_val = float(a_input)
    b_val = float(b_input)
    c_val = float(c_input)

    # Если цель - функция x^4 - 10x^2 + 9, то введите a=1, b=10, c=9.

    # Построение графика функции
    x_vals, f_vals = plot_function(a_val, b_val, c_val, n_val)

    # Поиск корня графическим методом
    approx_root_graph, index = find_graphical_root(x_vals, f_vals)
    if approx_root_graph is None:
        print("Смена знака не обнаружена на интервале. Графический метод не смог найти корень.")
        return
    else:
        print("Приблизительный корень, найденный графическим методом:", approx_root_graph)

    # Метод бисекции для уточнения корня
    x_left = x_vals[index]
    x_right = x_vals[index+1]
    approx_root_bis, iter_count = bisection_method(f, x_left, x_right, a_val, b_val, c_val)
    print("Корень, найденный методом бисекции:", approx_root_bis)
    print("Количество итераций в методе бисекции:", iter_count)

    # Абсолютная ошибка
    abs_error = abs(approx_root_graph - approx_root_bis)
    print("Абсолютная ошибка между графическим методом и методом бисекции:", abs_error)

if __name__ == "__main__":
    main()
