import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Чтение входных данных от пользователя
# ===============================
# Запрашиваем у пользователя нижнюю границу интервала [x, x+3]
lower_bound = float(input("Введите нижнюю границу интервала x (верхняя граница будет x+3): "))
upper_bound = lower_bound + 3  # интервал [x, x+3]

# Запрашиваем коэффициенты a, b, c для функции f(x)= x^3 - a*x^2 + b*x - c
a = float(input("Введите коэффициент a: "))
b = float(input("Введите коэффициент b: "))
c = float(input("Введите коэффициент c: "))

# Можно также задать значение допуска (tol) и максимальное число итераций (max_iter)
tol = 1e-10
max_iter = 1000

# ===============================
# Определение функции f(x)
# ===============================
def f(x):
    """Возвращает значение f(x)= x^3 - a*x^2 + b*x - c."""
    return x**3 - a*x**2 + b*x - c

# ===============================
# Вычисление точного решения (если возможно)
# ===============================
# Для кубического уравнения используем np.roots для получения всех корней.
# Коэффициенты полинома: [1, -a, b, -c]
roots = np.roots([1, -a, b, -c])
exact_root = None
# Выбираем действительный корень, лежащий в интервале [lower_bound, upper_bound]
for r in roots:
    if np.isreal(r):
        r_real = np.real(r)
        if lower_bound <= r_real <= upper_bound:
            exact_root = r_real
            break

if exact_root is None:
    print("\nНе удалось найти действительный корень, лежащий в интервале [{}, {}].".format(lower_bound, upper_bound))
    print("Возможно, в этом интервале корней нет или они комплексные.")
    # Можно продолжить работу с методами, но относительную погрешность рассчитать не получится.
else:
    print("\nТочный корень в интервале [{}, {}]: {:.12f}".format(lower_bound, upper_bound, exact_root))

# ===============================
# Реализация метода бисекции
# ===============================
def bisection_method(func, a_int, b_int, tol=1e-10, max_iter=1000):
    """
    Находит корень функции func на отрезке [a_int, b_int] методом бисекции.
    Возвращает приближённое значение корня и число итераций.
    """
    iterations = 0
    # Проверка условия наличия смены знака
    if func(a_int) * func(b_int) > 0:
        raise ValueError("На концах интервала нет смены знака. Метод бисекции неприменим.")

    while (b_int - a_int) > tol and iterations < max_iter:
        mid = (a_int + b_int) / 2.0
        f_mid = func(mid)
        if abs(f_mid) < tol:  # если значение функции очень близко к 0
            return mid, iterations
        # Определяем, в каком подинтервале происходит смена знака
        if func(a_int) * f_mid < 0:
            b_int = mid
        else:
            a_int = mid
        iterations += 1
    return (a_int + b_int) / 2.0, iterations

# ===============================
# Реализация метода Ньютона-Рафсона
# ===============================
def newton_raphson_method(func, dfunc, x0, tol=1e-10, max_iter=1000):
    """
    Находит корень функции func методом Ньютона-Рафсона,
    начиная с приближения x0.
    Возвращает приближённое значение корня и число итераций.
    """
    x = x0
    iterations = 0
    while iterations < max_iter:
        f_val = func(x)
        d_val = dfunc(x)
        if d_val == 0:
            raise ValueError("Произошло деление на ноль (производная равна 0).")
        x_new = x - f_val / d_val
        if abs(x_new - x) < tol:
            return x_new, iterations + 1
        x = x_new
        iterations += 1
    return x, iterations

# Определяем производную функции f(x)
def df(x):
    """Возвращает значение производной f'(x) для f(x)= x^3 - a*x^2 + b*x - c."""
    return 3*x**2 - 2*a*x + b

# ===============================
# Вычисление корней методами
# ===============================
try:
    # Метод бисекции
    root_bis, iter_bis = bisection_method(f, lower_bound, upper_bound, tol, max_iter)
    # Для метода Ньютона-Рафсона выберем начальное приближение как середину интервала
    initial_guess = (lower_bound + upper_bound) / 2.0
    root_nr, iter_nr = newton_raphson_method(f, df, initial_guess, tol, max_iter)
except ValueError as ve:
    print("\nОшибка:", ve)
    exit()

# Вычисление относительных погрешностей, если известен точный корень
if exact_root is not None and exact_root != 0:
    rel_error_bis = abs(root_bis - exact_root) / abs(exact_root)
    rel_error_nr = abs(root_nr - exact_root) / abs(exact_root)
else:
    rel_error_bis = rel_error_nr = None

# ===============================
# Вывод результатов
# ===============================
print("\nСравнение методов нахождения корня:")
print("-------------------------------------------")
print("{:<25} {:<20} {:<20}".format("Метод", "Число итераций", "Относительная погрешность"))
print("{:<25} {:<20} {:<20}".format("Метод бисекции", iter_bis,
                                    f"{rel_error_bis:.2e}" if rel_error_bis is not None else "N/A"))
print("{:<25} {:<20} {:<20}".format("Метод Ньютона-Рафсона", iter_nr,
                                    f"{rel_error_nr:.2e}" if rel_error_nr is not None else "N/A"))

print("\nНайденные корни:")
print("Корень (бисекция): {:.12f}".format(root_bis))
print("Корень (Ньютона-Рафсона): {:.12f}".format(root_nr))
if exact_root is not None:
    print("Точный корень: {:.12f}".format(exact_root))

# ===============================
# Построение графика функции и отмечание корней
# ===============================
# Для построения графика возьмём немного расширенный интервал для наглядности.
x_plot = np.linspace(lower_bound - 0.5, upper_bound + 0.5, 400)
y_plot = np.array([f(x) for x in x_plot])

plt.figure(figsize=(8, 6))
plt.plot(x_plot, y_plot, label=r'$f(x)=x^3 - {}x^2 + {}x - {}$'.format(a, b, c), color='blue')
plt.axhline(0, color='black', linewidth=0.5)

# Отмечаем найденные корни
plt.plot(root_bis, f(root_bis), 'ro', label='Корень (бисекция)')
plt.plot(root_nr, f(root_nr), 'go', label='Корень (Ньютона-Рафсона)')

# Если точное значение найдено, отмечаем его
if exact_root is not None:
    plt.plot(exact_root, f(exact_root), 'ks', label='Точный корень')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Нахождение корня в интервале [{:.2f}, {:.2f}]'.format(lower_bound, upper_bound))
plt.legend()
plt.grid(True)
plt.show()
