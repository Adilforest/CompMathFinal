from decimal import Decimal, InvalidOperation
import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QSplitter,
    QStackedWidget,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
)
from PySide6.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tasks.task1 import solve_task as solve_task_1
from tasks.task2 import solve_task as solve_task_2
from tasks.task3 import solve_task as solve_task_3
from tasks.task4 import solve_task as solve_task_4
from tasks.task5 import solve_task as solve_task_5
from tasks.task6 import solve_task as solve_task_6
from tasks.task7 import solve_task as solve_task_7
from tasks.task8 import solve_task as solve_task_8


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math Tasks App")
        self.resize(1100, 800)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        self.task_buttons = []
        for i in range(1, 9):
            btn = QPushButton(f"Task {i}")
            btn.clicked.connect(
                lambda checked, task=i: self.on_task_button_clicked(task)
            )
            self.task_buttons.append(btn)
            top_layout.addWidget(btn)
        main_layout.addWidget(top_panel)

        vertical_splitter = QSplitter(Qt.Vertical)

        horizontal_splitter = QSplitter(Qt.Horizontal)

        self.input_stack = QStackedWidget()
        self.plot_stack = QStackedWidget()
        self.console_stack = QStackedWidget()

        # Создаем страницы для каждой задачи (1–8)
        for i in range(8):
            task_number = i + 1
            if task_number == 1:
                input_widget = Task1InputWidget()
                input_widget.solveRequested.connect(self.solve_task1)
            elif task_number == 2:
                input_widget = Task2InputWidget()
                input_widget.solveRequested.connect(self.solve_task2)
            elif task_number == 3:
                input_widget = Task3InputWidget()
                input_widget.solveRequested.connect(self.solve_task_3)
            elif task_number == 4:
                input_widget = Task4InputWidget()
                input_widget.solveRequested.connect(self.solve_task4)
            elif task_number == 5:
                input_widget = Task5InputWidget()
                input_widget.solveRequested.connect(self.solve_task_5)
            elif task_number == 6:
                input_widget = Task6InputWidget()
                input_widget.solveRequested.connect(self.solve_task_6)
            elif task_number == 7:
                input_widget = Task7InputWidget()
                input_widget.solveRequested.connect(self.solve_task_7)
            elif task_number == 8:
                input_widget = Task8InputWidget()
                input_widget.solveRequested.connect(self.solve_task_8)
            else:
                # Placeholder for future tasks
                input_widget = QTextEdit()
                input_widget.setPlaceholderText(
                    f"Input and description for task {task_number}"
                )
            self.input_stack.addWidget(input_widget)

            # Для графика: создаем холст Matplotlib
            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            # Рисуем placeholder-график
            canvas.axes.text(
                0.5,
                0.5,
                f"Plot of the task {task_number}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=canvas.axes.transAxes,
            )
            canvas.draw()
            self.plot_stack.addWidget(canvas)

            # Для консоли: текстовое поле
            console = QTextEdit()
            console.setReadOnly(True)
            console.setPlaceholderText(f"Лог для задачи {task_number}")
            self.console_stack.addWidget(console)

        horizontal_splitter.addWidget(self.input_stack)
        horizontal_splitter.addWidget(self.plot_stack)
        horizontal_splitter.setStretchFactor(0, 1)
        horizontal_splitter.setStretchFactor(1, 2)

        vertical_splitter.addWidget(horizontal_splitter)
        vertical_splitter.addWidget(self.console_stack)
        vertical_splitter.setStretchFactor(0, 3)
        vertical_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(vertical_splitter)
        self.setCentralWidget(central_widget)

        # Изначально выбираем задачу 1
        self.current_task = 1
        self.on_task_button_clicked(1)

    def on_task_button_clicked(self, task):
        """Переключает видимые страницы для выбранной задачи."""
        self.current_task = task
        index = task - 1
        self.input_stack.setCurrentIndex(index)
        self.plot_stack.setCurrentIndex(index)
        self.console_stack.setCurrentIndex(index)

    def solve_task1(self, n_text, a_text, b_text, c_text):
        """Обработка нажатия кнопки Solve для задачи 1."""
        console = self.console_stack.widget(0)  # Консоль для задачи 1
        try:
            # Преобразуем входные данные
            n = Decimal(n_text)
            a = Decimal(a_text)
            b = -Decimal(b_text)
            c = Decimal(c_text)
        except InvalidOperation:
            console.append(
                "Error: invalid input. Please enter valid numbers for n, a, b, c."
            )
            return

        # Create the equation string for logging
        equation_str = f"f(x) = {a}x^4 - {b}x^2 + {c}"
        console.append(f"Solving equation: {equation_str} on interval [-{n}, {n}]")

        # Retrieve the Matplotlib canvas for the plot
        canvas = self.plot_stack.widget(0)
        # Execute the task
        results = solve_task_1(n, a, b, c, axes=canvas.axes)

        if results["approx_root_graph"] is None:
            console.append(
                "Graphical method did not find a root on the interval. No results to display."
            )
        else:
            console.append(
                f"Approximatly root (graphical method): {results['approx_root_graph']}"
            )
            console.append(
                f"Accurate root (bisection method, {results['iterations']} iterations): {results['approx_root_bis']}"
            )
            console.append(f"Absolute error: {results['abs_error']}")
        console.append("-" * 40)
        canvas.draw()

    def solve_task2(self, n1_text, n2_text, a_text, b_text, c_text, d_text, tol_text):
        """Обработка нажатия кнопки Solve для задачи 2."""
        console = self.console_stack.widget(1)
        try:
            n1 = Decimal(n1_text)
            n2 = Decimal(n2_text)
            a = Decimal(a_text)
            b = Decimal(b_text)
            c = Decimal(c_text)
            d = Decimal(d_text)
            tol = Decimal(tol_text)
        except InvalidOperation:
            console.append(
                "Error: invalid input. Please enter valid numbers for n1, n2, a, b, c, d, tol."
            )
            return

        # Создаём строку уравнения для логирования
        equation_str = f"f(x) = {a}x^3 + {b}x^2 + {c}x + {d}"
        console.append(f"Solving equation: {equation_str} on interval [{n1}, {n2}]")

        # Получаем объект холста для построения графика
        canvas = self.plot_stack.widget(1)

        # Вызываем функцию решения (solve_task_2) и получаем результаты
        results = solve_task_2(n1, n2, a, b, c, d, tol, axes=canvas.axes)

        # Выводим результаты в консоль:
        console.append(f"Exact root (via np.roots): {results['exact_root']}")
        console.append(
            f"Bisection method root: {results['bisection_root']} (Iterations: {results['bisection_iterations']})"
        )
        console.append(
            f"Newton-Raphson method root: {results['newton_root']} (Iterations: {results['newton_iterations']})"
        )
        # Выводим конечные абсолютные ошибки (последние значения в списках)
        final_bis_error = (
            results["bisection_absolute_errors"][-1]
            if results["bisection_absolute_errors"]
            else "N/A"
        )
        final_newton_error = (
            results["newton_absolute_errors"][-1]
            if results["newton_absolute_errors"]
            else "N/A"
        )
        console.append(f"Final absolute error (Bisection): {final_bis_error}")
        console.append(f"Final absolute error (Newton-Raphson): {final_newton_error}")
        console.append("-" * 40)

        # Обновляем график
        canvas.draw()

    def solve_task_3(self, omega_text, a_text, b_text, c_text):
        console = self.console_stack.widget(2)
        try:
            omega = float(omega_text)
            a = float(a_text)
            b = float(b_text)
            c = float(c_text)
        except ValueError:
            console.append(
                "Error: invalid input. Please enter valid numbers for omega, a, b, c."
            )
            return

        equation_str = f"z = b + c - a, x = b - z, y = c - z"
        console.append(f"Solving equation: {equation_str}")

        canvas = self.plot_stack.widget(2)
        results = solve_task_3(omega, a, b, c, axes=canvas.axes)

        console.append(f"Number of iterations: {results['iterations']}")
        console.append("Approximate solution using the relaxation method:")
        console.append(f"x = {results['x']}, y = {results['y']}, z = {results['z']}")

        canvas.draw()

    def solve_task4(self, matrix):
        console = self.console_stack.widget(3)
        try:
            matrix = np.array(matrix, dtype=float)
        except ValueError:
            console.append("Error: invalid matrix input.")
            return

        console.append("Solving power method task")
        console.append("Matrix A:")
        console.append(str(matrix))

        canvas = self.plot_stack.widget(3)
        results = solve_task_4(matrix, axes=canvas.axes)

        console.append(f"Approximate largest eigenvalue: {results['lambda_approx']}")
        console.append(f"Corresponding eigenvector (normalized):")
        console.append(str(results["eigenvector"]))
        console.append(f"Number of iterations: {len(results['iter_numbers'])}")
        console.append("-" * 40)

        canvas.draw()

    def solve_task_5(self, x_input, y_input):
        console = self.console_stack.widget(4)

        console.append("Solving exponential fit task")

        canvas = self.plot_stack.widget(4)
        results = solve_task_5(x_input, y_input, axes=canvas.axes)

        console.append("Comparison Table:")
        console.append(
            "{:<10} {:<15} {:<15} {:<15}".format("x", "y (data)", "y (model)", "Error")
        )
        for xi, yi, yi_fit, error in results["errors"]:
            console.append(
                "{:<10.4f} {:<15.4f} {:<15.4f} {:<15.4e}".format(xi, yi, yi_fit, error)
            )
        console.append(f"A fit: {results['A_fit']}")
        console.append(f"B fit: {results['B_fit']}")
        console.append(
            f"Equation: y = {results['A_fit']} * exp({results['B_fit']} * x)"
        )
        console.append("-" * 40)

        canvas.draw()

    def solve_task_6(self, x_input, y_input):
        console = self.console_stack.widget(5)

        console.append("Solving cubic spline interpolation task")

        canvas = self.plot_stack.widget(5)
        results = solve_task_6(x_input, y_input, axes=canvas.axes)

        console.append("Table of interpolated values:")
        console.append("{:<10} {:<15}".format("x", "Spline(y)"))
        for xi, yi in zip(results["x_table"], results["y_table"]):
            console.append("{:<10.4f} {:<15.4f}".format(xi, yi))
        console.append("-" * 40)

        canvas.draw()

    def solve_task_7(self, x_text, y0_text):
        console = self.console_stack.widget(6)
        try:
            x = float(x_text)
            y0 = float(y0_text)
        except ValueError:
            console.append(
                "Error: invalid input. Please enter valid numbers for x, y0."
            )
            return

        equation_str = f"dy/dx = x + y"
        console.append(f"Solving equation: {equation_str} at x = {x} with y(0) = {y0}")

        canvas = self.plot_stack.widget(6)
        results = solve_task_7(x, y0, axes=canvas.axes)

        console.append(f"Approximations:")
        for i, (approx_name, approx_val) in enumerate(results["approx_values"].items()):
            console.append(f"{approx_name}: {approx_val}")
        console.append(f"Final approximation: {results['final_approx']}")
        console.append("-" * 40)

        canvas.draw()

    def solve_task_8(self, n_text, a_text, b_text):
        console = self.console_stack.widget(7)
        try:
            n = int(n_text)
            a = float(a_text)
            b = float(b_text)
        except InvalidOperation:
            console.append(
                "Error: invalid input. Please enter valid numbers for n, a, b."
            )
            return

        equation_str = f"f(x) = sin(x)"
        console.append(f"Solving equation: {equation_str} on interval [{a}, {b}]")

        canvas = self.plot_stack.widget(7)
        results = solve_task_8(a, b, n, axes=canvas.axes)

        console.append(f"| {'x':^10} | {'f(x)':^10} |")
        for x, y in zip(results["x_vals"], results["f_vals"]):
            console.append(f"| {x:^10.4f} | {y:^10.4f} |")
        console.append(f"Approximation: {results['approx']}")
        console.append("-" * 40)

        canvas.draw()


class Task1InputWidget(QWidget):
    # Сигнал, который испускается при нажатии кнопки Solve.
    # Передаёт строки с параметрами n, a, b.
    solveRequested = Signal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.label_info = QLabel(
            """f(x) = ax^4 + b*x^2 + c
            
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_n = QLineEdit()
        self.input_n.setPlaceholderText("Enter n value (interval [-n, n])")
        layout.addWidget(self.input_n)

        self.input_a = QLineEdit()
        self.input_a.setPlaceholderText("Enter coefficient a")
        layout.addWidget(self.input_a)

        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("Enter coefficient b")
        layout.addWidget(self.input_b)

        self.input_c = QLineEdit()
        self.input_c.setPlaceholderText("Enter coefficient c")
        layout.addWidget(self.input_c)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        # Извлекаем тексты из полей ввода и испускаем сигнал
        n_text = self.input_n.text().strip()
        a_text = self.input_a.text().strip()
        b_text = self.input_b.text().strip()
        c_text = self.input_c.text().strip()
        self.solveRequested.emit(n_text, a_text, b_text, c_text)


class Task2InputWidget(QWidget):
    solveRequested = Signal(str, str, str, str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """f(x) = ax^3 + b*x^2 + cx + d
            
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_n1 = QLineEdit()
        self.input_n1.setPlaceholderText("Enter n1 value (interval [n1, n2])")
        layout.addWidget(self.input_n1)

        self.input_n2 = QLineEdit()
        self.input_n2.setPlaceholderText("Enter n2 value (interval [n1, n2])")
        layout.addWidget(self.input_n2)

        self.input_a = QLineEdit()
        self.input_a.setPlaceholderText("Enter coefficient a")
        layout.addWidget(self.input_a)

        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("Enter coefficient b")
        layout.addWidget(self.input_b)

        self.input_c = QLineEdit()
        self.input_c.setPlaceholderText("Enter coefficient c")
        layout.addWidget(self.input_c)

        self.input_d = QLineEdit()
        self.input_d.setPlaceholderText("Enter coefficient d")
        layout.addWidget(self.input_d)

        self.input_tol = QLineEdit(text="1e-10")
        self.input_tol.setPlaceholderText("Enter tolerance value (1e-10)")
        layout.addWidget(self.input_tol)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        n1_text = self.input_n1.text().strip()
        n2_text = self.input_n2.text().strip()
        a_text = self.input_a.text().strip()
        b_text = self.input_b.text().strip()
        c_text = self.input_c.text().strip()
        d_text = self.input_d.text().strip()
        tol_text = self.input_tol.text().strip()
        self.solveRequested.emit(
            n1_text, n2_text, a_text, b_text, c_text, d_text, tol_text
        )


class Task3InputWidget(QWidget):
    solveRequested = Signal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """Relaxation method for solving the system of equations
The "exact" relations are:
z = b + c - a
x = b - z
y = c - z
            """
        )
        layout.addWidget(self.label_info)

        self.input_omega = QLineEdit()
        self.input_omega.setPlaceholderText("Enter the relaxation parameter ω")
        layout.addWidget(self.input_omega)

        self.input_a = QLineEdit()
        self.input_a.setPlaceholderText("Enter the value of a")
        layout.addWidget(self.input_a)

        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("Enter the value of b")
        layout.addWidget(self.input_b)

        self.input_c = QLineEdit()
        self.input_c.setPlaceholderText("Enter the value of c")
        layout.addWidget(self.input_c)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        omega_text = self.input_omega.text().strip()
        a_text = self.input_a.text().strip()
        b_text = self.input_b.text().strip()
        c_text = self.input_c.text().strip()
        self.solveRequested.emit(omega_text, a_text, b_text, c_text)


class Task4InputWidget(QWidget):
    solveRequested = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """Power method for finding the largest eigenvalue of a matrix A
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_size = QLineEdit(text="3")
        self.input_size.setPlaceholderText("Enter the size of the square matrix")
        layout.addWidget(self.input_size)

        self.input_matrix = QTableWidget()
        self.input_matrix.setColumnCount(3)
        self.input_matrix.setRowCount(3)
        matrix = [
            [6, 2, 3],
            [2, 6, 4],
            [3, 4, 6],
        ]
        for i in range(3):
            for j in range(3):
                self.input_matrix.setItem(i, j, QTableWidgetItem(str(matrix[i][j])))
        self.input_matrix.resizeColumnsToContents()
        layout.addWidget(self.input_matrix)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)
        self.input_size.textChanged.connect(self.on_size_changed)

    def on_solve_clicked(self):
        size = self.input_matrix.rowCount()
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                item = self.input_matrix.item(i, j)
                if item is None:
                    return
                try:
                    matrix[i, j] = float(item.text())
                except ValueError:
                    return
        self.solveRequested.emit(matrix)

    def on_size_changed(self):
        try:
            size = int(self.input_size.text())
            self.input_matrix.setColumnCount(size)
            self.input_matrix.setRowCount(size)
            self.input_matrix.setVerticalHeaderLabels([f"{i}" for i in range(size)])
            self.input_matrix.resizeColumnsToContents()
        except ValueError:
            pass


class Task5InputWidget(QWidget):
    solveRequested = Signal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """Exponential fit: y = A * exp(B * x)
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_x = QLineEdit()
        self.input_x.setPlaceholderText("Enter x values separated by spaces")
        layout.addWidget(self.input_x)

        self.input_y = QLineEdit()
        self.input_y.setPlaceholderText("Enter y values separated by spaces")
        layout.addWidget(self.input_y)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        x_text = self.input_x.text().strip()
        y_text = self.input_y.text().strip()
        x_input = np.array([float(x) for x in x_text.split()])
        y_input = np.array([float(y) for y in y_text.split()])
        self.solveRequested.emit(x_input, y_input)


class Task6InputWidget(QWidget):
    solveRequested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """Cubic spline interpolation
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_x = QLineEdit()
        self.input_x.setPlaceholderText("Enter x values separated by spaces")
        layout.addWidget(self.input_x)

        self.input_y = QLineEdit()
        self.input_y.setPlaceholderText("Enter y values separated by spaces")
        layout.addWidget(self.input_y)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        x_text = self.input_x.text().strip()
        y_text = self.input_y.text().strip()
        self.solveRequested.emit(x_text, y_text)


class Task7InputWidget(QWidget):
    solveRequested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """dy/dx = x + y
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_x = QLineEdit()
        self.input_x.setPlaceholderText("Enter the value of x to compute y")
        layout.addWidget(self.input_x)

        self.input_y0 = QLineEdit(text="1.0")
        self.input_y0.setPlaceholderText("Enter the initial value y(0)")
        layout.addWidget(self.input_y0)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        x_text = self.input_x.text().strip()
        y0_text = self.input_y0.text().strip()
        self.solveRequested.emit(x_text, y0_text)


class Task8InputWidget(QWidget):
    solveRequested = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label_info = QLabel(
            """f(x) = sin(x)
            
            """
        )
        layout.addWidget(self.label_info)

        self.input_n = QLineEdit()
        self.input_n.setPlaceholderText("Enter the number of subintervals")
        layout.addWidget(self.input_n)

        self.input_a = QLineEdit()
        self.input_a.setPlaceholderText("Enter the lower bound of integration")
        layout.addWidget(self.input_a)

        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("Enter the upper bound of integration")
        layout.addWidget(self.input_b)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        n_text = self.input_n.text().strip()
        a_text = self.input_a.text().strip()
        b_text = self.input_b.text().strip()
        self.solveRequested.emit(n_text, a_text, b_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
