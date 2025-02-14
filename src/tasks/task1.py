import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


def solve_task1(n, a, b, axes=None):
    """
    Решает задачу 1:
      Вычисляет функцию f(x) = x^4 - a*x^2 + b на интервале [-n, n],
      ищет приближённый корень (графическим методом и методом бисекции),
      строит график функции и возвращает результаты в виде словаря.

    Параметры:
      n, a, b: значения (можно передавать как Decimal, str или float).
      axes: объект matplotlib.axes, на котором нужно нарисовать график.
             Если не передан, создаётся новое окно.

    Возвращает:
      dict с ключами:
         - "x_vals": массив значений x,
         - "f_vals": массив значений f(x),
         - "approx_root_graph": приближённый корень, найденный графическим методом (или None, если не найден),
         - "approx_root_bis": уточнённый корень методом бисекции (или None),
         - "abs_error": абсолютная ошибка между методами (или None),
         - "iterations": число итераций метода бисекции (или None).
    """
    # Преобразуем входные данные к float
    n_val = float(n)
    a_val = float(a)
    b_val = float(b)

    # Определяем функцию f(x)
    def f(x):
        return x**4 - a_val * x**2 + b_val

    # Создаём массив значений x и соответствующих f(x)
    x_vals = np.linspace(-n_val, n_val, 1000)
    f_vals = np.array([f(x) for x in x_vals])

    # Шаг 5. Поиск приближённого корня графическим методом (смена знака)
    approx_root_graph = None
    sign_change_index = None
    for i in range(len(x_vals) - 1):
        if f_vals[i] * f_vals[i + 1] < 0:
            approx_root_graph = (x_vals[i] + x_vals[i + 1]) / 2.0
            sign_change_index = i
            break

    # Если смена знака не обнаружена, корень не найден
    if approx_root_graph is None:
        approx_root_bis = None
        abs_error = None
        iterations = None
    else:
        # Шаг 6. Метод бисекции для уточнения корня
        x_left = x_vals[sign_change_index]
        x_right = x_vals[sign_change_index + 1]
        tol = 1e-10
        max_iter = 1000
        iterations = 0

        while (x_right - x_left) > tol and iterations < max_iter:
            x_mid = (x_left + x_right) / 2.0
            f_mid = f(x_mid)
            if abs(f_mid) < tol:
                break
            # Определяем, в какой половине происходит смена знака
            if f(x_left) * f_mid < 0:
                x_right = x_mid
            else:
                x_left = x_mid
            iterations += 1

        approx_root_bis = (x_left + x_right) / 2.0
        abs_error = abs(approx_root_graph - approx_root_bis)

    # Шаг 4. Построение графика
    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 6))
    else:
        axes.clear()

    axes.plot(x_vals, f_vals, label=f"f(x) = x⁴ - {a_val}x² + {b_val}")
    axes.axhline(0, color="black", linewidth=0.5)  # ось x
    axes.axvline(0, color="black", linewidth=0.5)  # ось y
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.set_title(f"График функции на интервале [-{n_val}, {n_val}]")
    axes.legend()
    axes.grid(True)

    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    # Возвращаем результаты в виде словаря
    return {
        "x_vals": x_vals,
        "f_vals": f_vals,
        "approx_root_graph": approx_root_graph,
        "approx_root_bis": approx_root_bis,
        "abs_error": abs_error,
        "iterations": iterations,
    }


# Пример использования функции в консольном режиме
if __name__ == "__main__":
    # Динамический ввод параметров
    n = Decimal(input("Введите значение n (интервал будет [-n, n]): "))
    a = Decimal(input("Введите коэффициент a: "))
    b = Decimal(input("Введите коэффициент b: "))

    # Вызываем функцию (график будет нарисован в новом окне)
    results = solve_task1(n, a, b)

    # Выводим результаты в консоль
    if results["approx_root_graph"] is None:
        print(
            "Смена знака не обнаружена на интервале. Графический метод не смог найти корень."
        )
    else:
        print(
            "Приблизительный корень, найденный графическим методом:",
            results["approx_root_graph"],
        )
        print("Корень, найденный методом бисекции:", results["approx_root_bis"])
        print("Абсолютная ошибка между методами:", results["abs_error"])

    # Отображаем график (если axes не был передан)
    plt.show()
