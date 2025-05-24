import tkinter as tk
import random
from tkinter import filedialog, messagebox, ttk
import numpy as np


class InputVectors:
    MAX_MATRIX_SIZE = 10

    # <summary>
    # Ініціалізує екземпляр InputVectors з посиланням на головне вікно Tkinter.
    # </summary>
    # <param name="root" type="tk.Tk">Головне вікно Tkinter</param>
    def __init__(self, root: tk.Tk):
        self.root = root

    # <summary>
    # Відображає вікно діалогове для ручного введення матриці із заданими розмірами.
    # Перевіряє правильність введення і повертає матрицю як NumPy-масив.
    # </summary>
    # <param name="title" type="str">Заголовок діалогу введення</param>
    # <param name="rows" type="int">Кількість рядків у матриці</param>
    # <param name="cols" type="int">Кількість стовпців у матриці</param>
    # <returns type="np.ndarray">Матриця як NumPy-масив або порожній масив у разі помилки</returns>
    def input_matrix_gui(
            self, title: str, rows: int, cols: int, precision: int
    ) -> np.ndarray:
        if rows > self.MAX_MATRIX_SIZE or cols > self.MAX_MATRIX_SIZE:
            messagebox.showerror(
                "Error",
                f"Maximum dimension for manual input is {self.MAX_MATRIX_SIZE}",
            )
            return np.array([])

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.grab_set()
        dialog.resizable(False, False)

        entry_width = 8
        label_width = 6
        padding = 10
        button_height = 40

        window_width = min(max(cols * (entry_width * 10 + label_width * 15 + 15 * padding), 300), 1200)
        window_height = min(max(rows * 30 + button_height + 60, 200), 600)
        dialog.geometry(f"{int(window_width)}x{int(window_height)}")

        content_frame = ttk.Frame(dialog)
        content_frame.pack(padx=10, pady=10)

        entries = [
            [ttk.Entry(content_frame, width=entry_width) for _ in range(cols)]
            for _ in range(rows)
        ]

        for i in range(rows):
            for j in range(cols):
                ttk.Label(content_frame, text=f"[{i + 1},{j + 1}]").grid(
                    row=i, column=j * 2, padx=2, pady=2
                )
                entries[i][j].grid(row=i, column=j * 2 + 1, padx=2, pady=2)

        button_frame = ttk.Frame(dialog)
        button_frame.place(relx=0, rely=1.0, x=20, y=-60, anchor="sw")

        result = []

        # <summary>
        # Зчитує введені значення з полів, перевіряє їх на коректність,
        # формує матрицю та закриває діалогове вікно.
        # </summary>
        def submit():
            nonlocal result
            matrix = []
            for count_row, row_entries in enumerate(entries):
                row = []
                for count_col, entry in enumerate(row_entries):
                    try:
                        row.append(float(entry.get().strip()))
                    except ValueError:
                        messagebox.showerror(
                            "Error", f"Enter valid numbers [{count_row + 1},{count_col + 1}]"
                        )
                        return
                matrix.append(row)
            result = matrix
            dialog.destroy()

        ttk.Button(button_frame, text="Submit", command=submit).pack(side="left", padx=5)
        dialog.wait_window()
        return np.round(np.array(result), precision)

    # <summary>
    # Генерує випадкову матрицю з заданими розмірами, діапазоном значень і точністю округлення.
    # </summary>
    # <param name="min_val" type="float">Мінімальне значення</param>
    # <param name="max_val" type="float">Максимальне значення</param>
    # <param name="rows" type="int">Кількість рядків</param>
    # <param name="cols" type="int">Кількість стовпців</param>
    # <param name="precision" type="int">Кількість знаків після коми для округлення</param>
    # <returns type="np.ndarray">Згенерована та округлена матриця</returns>
    @staticmethod
    def generate_random_matrix(
            min_val: float, max_val: float, rows: int, cols: int, precision: int
    ) -> np.ndarray:
        mat = np.random.uniform(min_val, max_val, size=(rows, cols))
        return np.round(mat, precision)

    # <summary>
    # Відкриває діалог для завантаження матриці з текстового файлу з роздільниками-комами.
    # </summary>
    # <returns type="np.ndarray | None">Завантажена матриця або порожній масив, якщо скасовано або коректно задано</returns>
    @staticmethod
    def load_from_file() -> np.ndarray | None:
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return np.array([])
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=float)
            return data
        except ValueError:
            messagebox.showerror("Error", "Invalid numbers in file")
