import tkinter as tk
import random
from tkinter import filedialog, messagebox, ttk
import numpy as np


class InputVectors:
    MAX_MATRIX_SIZE = 10
    MAX_PRECISION = 9
    MIN_PRECISION = 0

    def __init__(self, root: tk.Tk):
        self.root = root

    def input_matrix_gui(self, title: str, rows: int, cols: int) -> np.ndarray:
        if (
            rows > self.MAX_MATRIX_SIZE
            or cols > self.MAX_MATRIX_SIZE
        ):
            messagebox.showerror(
                "Error",
                f"Maximum dimension for manual input is {self.MAX_MATRIX_SIZE}",
            )
            return np.array([])

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.grab_set()
        dialog.resizable(True, True)

        entry_width = 8
        label_width = 6
        padding = 10
        button_height = 40
        window_width = min(
            max(cols * (entry_width * 10 + label_width * 10 + padding) + 40, 300), 800
        )
        window_height = min(max(rows * 40 + button_height + 60, 200), 600)
        dialog.geometry(f"{int(window_width)}x{int(window_height)}")

        canvas = tk.Canvas(dialog)
        scrollbar_y = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(dialog, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        canvas_frame = canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw"
        )

        entries = [
            [ttk.Entry(scrollable_frame, width=entry_width) for _ in range(cols)]
            for _ in range(rows)
        ]

        for i in range(rows):
            for j in range(cols):
                ttk.Label(scrollable_frame, text=f"[{i + 1},{j + 1}]").grid(
                    row=i, column=j * 2, padx=2, pady=2
                )
                entries[i][j].grid(row=i, column=j * 2 + 1, padx=2, pady=2)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        result = []

        def submit():
            nonlocal result
            matrix = []
            for count_row, row_entries in enumerate(entries):
                row = []
                for count_col, entry in enumerate(row_entries):
                    try:
                        # if '.' is float(entry.get().strip()):
                        #     decimals = s.split('.')[1]
                        #     if len(decimals.rstrip('0')) > self.MAX_PRECISION:
                        #         messagebox.showerror(
                        #             "Error", f"Precision must be between {self.MIN_PRECISION} and {self.MAX_PRECISION}"
                        #         )
                        #         return
                        row.append(float(entry.get().strip()))
                    except ValueError:
                        messagebox.showerror(
                            "Error", f"Enter valid numbers {count_row + 1}{count_col + 1}"
                        )
                        return
                matrix.append(row)
            result = matrix
            dialog.destroy()

        ttk.Button(button_frame, text="Submit", command=submit).pack(
            side="left", padx=5
        )

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_frame, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)
        dialog.wait_window()
        return np.array(result)

    @staticmethod
    def generate_random_matrix(
            min_val: float, max_val: float, rows: int, cols: int, precision: int
    ) -> np.ndarray:
        return np.array(
            [
                [
                    round(random.uniform(min_val, max_val), precision)
                    for _ in range(cols)
                ]
                for _ in range(rows)
            ]
        )

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
