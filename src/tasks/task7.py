import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Определяем символьные переменные
x, t = sp.symbols('x t')

# ===============================
# Шаг 1. Определяем начальное приближение
# ===============================
# Начальное приближение y0(x) = 1 (константа, т.к. y(0)=1)
y0 = sp.sympify(1)

# Сохраняем аппроксимации в список
picard_approximations = [y0]

# ===============================
# Шаг 2. Вычисляем последовательно аппроксимации по схеме Пикара
# ===============================
# Метод Пикара: y_{n+1}(x) = 1 + ∫[0,x] (t + y_n(t)) dt
for i in range(1, 5):
    # Получаем предыдущую аппроксимацию y_{n}(t)
    y_prev = picard_approximations[i - 1]
    # Интегрируем функцию: t + y_prev(t)
    integrand = t + y_prev.subs(x, t)
    y_new = 1 + sp.integrate(integrand, (t, 0, x))
    y_new = sp.simplify(y_new)
    picard_approximations.append(y_new)

# ===============================
# Шаг 3. Вывод аналитических выражений аппроксимаций
# ===============================
print("Аппроксимации методом Пикара:")
for i, yi in enumerate(picard_approximations):
    print(f"y_{i}(x) =")
    sp.pprint(yi)
    print("\n")

# ===============================
# Шаг 4. Вычисление значений аппроксимаций при x = 0.2 и вывод таблицы
# ===============================
x_val = 0.2
print("Таблица значений y(x) при x = 0.2:")
print("{:<12} {:<30}".format("Итерация", "y(x)"))
for i, yi in enumerate(picard_approximations):
    yi_val = yi.subs(x, x_val).evalf()
    print("{:<12} {:<30.6f}".format(i, yi_val))

# Для четвертой аппроксимации:
y4_val = picard_approximations[4].subs(x, x_val).evalf()
print(f"\nЧетвертая аппроксимация: y_4({x_val}) = {y4_val:.6f}")

# ===============================
# Шаг 5. Построение графика аппроксимаций
# ===============================
# Генерируем набор значений x в интервале [0, 0.3]
x_vals = np.linspace(0, 0.3, 200)
plt.figure(figsize=(10, 6))
for i, yi in enumerate(picard_approximations):
    # Преобразуем символьное выражение в функцию для numpy
    f_yi = sp.lambdify(x, yi, 'numpy')
    y_vals = f_yi(x_vals)
    # Если результат скаляр, создаём массив той же формы, заполненный этим значением
    if np.ndim(y_vals) == 0:
        y_vals = np.full_like(x_vals, y_vals)
    plt.plot(x_vals, y_vals, label=f'$y_{i}(x)$')

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title("Аппроксимации методом Пикара для $\dfrac{dy}{dx}=x+y$, $y(0)=1$")
plt.legend()
plt.grid(True)
plt.show()
