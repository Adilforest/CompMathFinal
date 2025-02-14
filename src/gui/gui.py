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

from tasks.task1 import solve_task1


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
            else:
                # Для остальных задач пока используем placeholder
                input_widget = QTextEdit()
                input_widget.setPlaceholderText(
                    f"Ввод и описание для задачи {task_number}"
                )
            self.input_stack.addWidget(input_widget)

            # Для графика: создаем холст Matplotlib
            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            # Рисуем placeholder-график
            canvas.axes.text(
                0.5,
                0.5,
                f"График задачи {task_number}",
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

        console = self.console_stack.currentWidget()
        console.append(f"Переключение на задачу {task}")

    def solve_task1(self, n_text, a_text, b_text):
        """Обработка нажатия кнопки Solve для задачи 1."""
        console = self.console_stack.widget(0)  # Консоль для задачи 1
        try:
            # Преобразуем входные данные
            n = Decimal(n_text)
            a = Decimal(a_text)
            b = Decimal(b_text)
        except InvalidOperation:
            console.append("Ошибка: введите корректные числовые значения для n, a и b.")
            return

        # Формируем строку уравнения для лога
        equation_str = f"f(x) = x^4 - {a}x^2 + {b}"
        console.append(f"Solving equation: {equation_str} on interval [-{n}, {n}]")

        # Получаем холст для графика задачи 1
        canvas = self.plot_stack.widget(0)
        # Вызываем функцию решения задачи, передавая ось для рисования графика
        results = solve_task1(n, a, b, axes=canvas.axes)

        if results["approx_root_graph"] is None:
            console.append(
                "Графический метод: смена знака не обнаружена. Корень не найден."
            )
        else:
            console.append(
                f"Приблизительный корень (графический метод): {results['approx_root_graph']}"
            )
            console.append(
                f"Уточнённый корень (метод бисекции, {results['iterations']} итераций): {results['approx_root_bis']}"
            )
            console.append(f"Абсолютная ошибка: {results['abs_error']}")
        console.append("-" * 40)
        canvas.draw()


class Task1InputWidget(QWidget):
    # Сигнал, который испускается при нажатии кнопки Solve.
    # Передаёт строки с параметрами n, a, b.
    solveRequested = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.label_info = QLabel(
            "Введите параметры для уравнения f(x) = x^4 - a*x^2 + b"
        )
        layout.addWidget(self.label_info)

        self.input_n = QLineEdit()
        self.input_n.setPlaceholderText("Введите значение n (интервал [-n, n])")
        layout.addWidget(self.input_n)

        self.input_a = QLineEdit()
        self.input_a.setPlaceholderText("Введите коэффициент a")
        layout.addWidget(self.input_a)

        self.input_b = QLineEdit()
        self.input_b.setPlaceholderText("Введите коэффициент b")
        layout.addWidget(self.input_b)

        self.solve_button = QPushButton("Solve")
        layout.addWidget(self.solve_button)

        self.solve_button.clicked.connect(self.on_solve_clicked)

    def on_solve_clicked(self):
        # Извлекаем тексты из полей ввода и испускаем сигнал
        n_text = self.input_n.text().strip()
        a_text = self.input_a.text().strip()
        b_text = self.input_b.text().strip()
        self.solveRequested.emit(n_text, a_text, b_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
