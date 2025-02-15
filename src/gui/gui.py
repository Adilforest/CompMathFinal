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
)
from PySide6.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tasks.task1 import solve_task as solve_task_1
from tasks.task2 import solve_task as solve_task_2
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

        # Верхняя панель с кнопками задач (фиксированный размер)
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

        # Вертикальный сплиттер: верхняя часть (ввод/график) и нижняя (консоль)
        vertical_splitter = QSplitter(Qt.Vertical)

        # Горизонтальный сплиттер: левая часть (ввод) и правая (график)
        horizontal_splitter = QSplitter(Qt.Horizontal)

        # QStackedWidget для ввода данных/описания
        self.input_stack = QStackedWidget()
        # QStackedWidget для графиков (Matplotlib)
        self.plot_stack = QStackedWidget()
        # QStackedWidget для консольного лога
        self.console_stack = QStackedWidget()

        # Создаем страницы для каждой задачи (1–8)
        for i in range(8):
            task_number = i + 1
            if task_number == 1:
                # Для первой задачи используем наш кастомный виджет ввода
                input_widget = Task1InputWidget()
                # Связываем сигнал solveRequested с обработчиком задачи 1
                input_widget.solveRequested.connect(self.solve_task1)
            elif task_number == 2:
                # Для второй задачи используем другой кастомный виджет ввода
                input_widget = Task2InputWidget()
                # Связываем сигнал solveRequested с обработчиком задачи 2
                input_widget.solveRequested.connect(self.solve_task2)
            elif task_number == 7:
                # Для седьмой задачи используем еще один кастомный виджет ввода
                input_widget = Task7InputWidget()
                # Связываем сигнал solveRequested с обработчиком задачи 7
                input_widget.solveRequested.connect(self.solve_task_7)
            elif task_number == 8:
                # Для восьмой задачи используем еще один кастомный виджет ввода
                input_widget = Task8InputWidget()
                # Связываем сигнал solveRequested с обработчиком задачи 8
                input_widget.solveRequested.connect(self.solve_task_8)
            else:
                # Для остальных задач пока используем placeholder
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
            b = Decimal(b_text)
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

        equation_str = f"f(x) = {a}x^3 + {b}x^2 + {c}x + {d}"
        console.append(f"Solving equation: {equation_str} on interval [{n1}, {n2}]")

        canvas = self.plot_stack.widget(1)
        results = solve_task_2(n1, n2, a, b, c, d, tol, axes=canvas.axes)

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
            """f(x) = x^4 - a*x^2 + b
            
            
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
