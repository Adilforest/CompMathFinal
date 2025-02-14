from decimal import Decimal
import matplotlib.pyplot as plt

# ===============================
# Шаг 1. Ввод параметров
# ===============================
# Вводим релаксационный параметр ω и коэффициенты a, b, c
omega = Decimal(input("Введите значение релаксационного параметра ω: "))
a = Decimal(input("Введите значение a: "))
b = Decimal(input("Введите значение b: "))
c = Decimal(input("Введите значение c: "))

# Преобразуем Decimal в float для удобства вычислений
omega_val = float(omega)
a_val = float(a)
b_val = float(b)
c_val = float(c)

# ===============================
# Шаг 2. Задаём параметры итерационного процесса
# ===============================
tol = 1e-10         # Допуск по изменениям
max_iter = 1000     # Максимальное число итераций

# Начальные приближения (можно задать произвольные значения)
x = 0.0
y = 0.0
z = 0.0

# Для хранения истории итераций для построения графика
iterations_list = []
x_history = []
y_history = []
z_history = []

# ===============================
# Шаг 3. Итерационный процесс по методу релаксации
# ===============================
for iter in range(max_iter):
    # Сохраняем предыдущие значения для проверки сходимости
    x_old, y_old, z_old = x, y, z

    # Обновление z по релаксационному правилу:
    # "точное" соотношение: z = b + c - a
    z = (1 - omega_val) * z_old + omega_val * (b_val + c_val - a_val)

    # Обновление x по уравнению: x = b - z (используем новое z)
    x = (1 - omega_val) * x_old + omega_val * (b_val - z)

    # Обновление y по уравнению: y = c - z (используем новое z)
    y = (1 - omega_val) * y_old + omega_val * (c_val - z)

    # Сохраняем значения для графика
    iterations_list.append(iter)
    x_history.append(x)
    y_history.append(y)
    z_history.append(z)

    # Проверяем условие сходимости: если максимальное изменение меньше tol, останавливаемся
    if max(abs(x - x_old), abs(y - y_old), abs(z - z_old)) < tol:
        break

# ===============================
# Шаг 4. Вывод результатов
# ===============================
print("\nКоличество итераций:", iter + 1)
print("Приближённое решение системы методом релаксации:")
print(f"x = {x:.12f}")
print(f"y = {y:.12f}")
print(f"z = {z:.12f}")

# Для сравнения выводим аналитическое решение:
exact_x = a_val - c_val
exact_y = a_val - b_val
exact_z = b_val + c_val - a_val

print("\nАналитическое решение:")
print(f"x = {exact_x:.12f}")
print(f"y = {exact_y:.12f}")
print(f"z = {exact_z:.12f}")

# ===============================
# Шаг 5. Построение графика сходимости
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(iterations_list, x_history, label='x', marker='o', markersize=4)
plt.plot(iterations_list, y_history, label='y', marker='s', markersize=4)
plt.plot(iterations_list, z_history, label='z', marker='^', markersize=4)

# Добавляем горизонтальные линии с аналитическими решениями для сравнения
plt.axhline(exact_x, color='blue', linestyle='--', linewidth=1)
plt.axhline(exact_y, color='orange', linestyle='--', linewidth=1)
plt.axhline(exact_z, color='green', linestyle='--', linewidth=1)

plt.xlabel('Итерация')
plt.ylabel('Значение переменной')
plt.title('Сходимость переменных методом релаксации')
plt.legend()
plt.grid(True)
plt.show()
