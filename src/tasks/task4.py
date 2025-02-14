import numpy as np
import matplotlib.pyplot as plt

def power_method(A, tol=1e-10, max_iter=1000):
    """
    Реализует метод степенных итераций для нахождения
    наибольшего по модулю собственного значения матрицы A.

    Параметры:
    - A: квадратная матрица (numpy.ndarray)
    - tol: заданный допуск (конвергенция по изменению собственного значения)
    - max_iter: максимальное число итераций

    Возвращает:
    - lambda_approx: приближённое наибольшее собственное значение
    - x: соответствующий собственный вектор (нормированный)
    - eigenvalue_history: список приближений собственного значения по итерациям
    - iter_numbers: список номеров итераций (для построения графика)
    """
    n = A.shape[0]
    # Начальное приближение: произвольный вектор (здесь - вектор единиц)
    x = np.ones(n)
    x = x / np.linalg.norm(x)  # нормируем вектор
    eigenvalue_history = []
    iter_numbers = []

    for i in range(max_iter):
        # Вычисляем произведение A * x
        y = A.dot(x)
        # Приближение собственного значения вычисляем по формуле Релея:
        lambda_approx = np.dot(x, y)
        eigenvalue_history.append(lambda_approx)
        iter_numbers.append(i)

        # Нормируем полученный вектор y для получения следующего приближения собственного вектора
        x_new = y / np.linalg.norm(y)

        # Проверка сходимости: если разность между последовательными приближениями меньше tol, выходим
        if i > 0 and abs(eigenvalue_history[-1] - eigenvalue_history[-2]) < tol:
            break

        x = x_new

    return lambda_approx, x, eigenvalue_history, iter_numbers

# ===============================
# Чтение матрицы A от пользователя
# ===============================
print("Введите элементы матрицы A (3x3):")
a11 = float(input("a11: "))
a12 = float(input("a12: "))
a13 = float(input("a13: "))
a21 = float(input("a21: "))
a22 = float(input("a22: "))
a23 = float(input("a23: "))
a31 = float(input("a31: "))
a32 = float(input("a32: "))
a33 = float(input("a33: "))

A = np.array([[a11, a12, a13],
              [a21, a22, a23],
              [a31, a32, a33]])

# Если хотите использовать тестовый пример, раскомментируйте следующую строку:
# A = np.array([[6, 2, 3],
#               [2, 6, 4],
#               [3, 4, 6]])

# ===============================
# Нахождение наибольшего собственного значения методом степенных итераций
# ===============================
lambda_approx, eigenvector, eigenvalue_history, iter_numbers = power_method(A)

# Вывод результатов
print("\nРезультаты метода степенных итераций:")
print("Приближённое наибольшее собственное значение:", lambda_approx)
print("Соответствующий собственный вектор (нормированный):")
print(eigenvector)
print("Число итераций:", len(iter_numbers))

# ===============================
# Вывод таблицы сходимости
# ===============================
print("\nТаблица сходимости (итерация, приближение собственного значения):")
print("{:<10} {:<20}".format("Итерация", "Собственное значение"))
for it, val in zip(iter_numbers, eigenvalue_history):
    print("{:<10} {:<20.12f}".format(it, val))

# ===============================
# Построение графика сходимости
# ===============================
plt.figure(figsize=(8, 6))
plt.plot(iter_numbers, eigenvalue_history, marker='o', linestyle='-', color='blue')
plt.xlabel("Номер итерации")
plt.ylabel("Приближение собственного значения")
plt.title("Сходимость метода степенных итераций")
plt.grid(True)
plt.show()
