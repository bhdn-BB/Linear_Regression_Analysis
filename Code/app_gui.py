import tkinter as tk
from tkinter import ttk
import numpy as np


class AppGui:
    def __init__(self, root: tk.Tk, app):
        self.root = root
        self.app = app
        self.setup_gui()

    def setup_gui(self):

        self.root.title("Linear Regression Analysis")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        taskbar_height = 100
        window_height = screen_height - taskbar_height
        self.root.geometry(f"{screen_width}x{window_height}+0+0")

        self.root.option_add("*Font", "Arial 10")
        self.root.resizable(False, False)
        self.root.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure((1, 2, 3, 4), weight=1)

        self.setup_title_label()
        self.setup_dimensions_panel()
        self.setup_input_panels()
        self.setup_results_panel()

    def setup_title_label(self):
        label = ttk.Label(self.root, text="Welcome to the Linear Regression Analysis App", font=("Arial", 14, "bold"), anchor="center")
        label.grid(row=0, column=0, columnspan=5, padx=5, pady=2, sticky="ew")

    def setup_dimensions_panel(self):
        frame = ttk.LabelFrame(
            self.root,
            text=f"Dataset dimensions are within the range[{self.app.MIN_DIMENSION_DATASET};{self.app.MAX_DIMENSION_DATASET}]",
        )
        frame.grid(row=1, column=0, columnspan=5, padx=5, pady=2, sticky="ew")

        ttk.Label(frame, text="Observations:").grid(row=0, column=0, padx=2, pady=2)
        self.obs_entry = ttk.Entry(frame, width=10)
        self.obs_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(frame, text="Features:").grid(row=0, column=2, padx=2, pady=2)
        self.feat_entry = ttk.Entry(frame, width=10)
        self.feat_entry.grid(row=0, column=3, padx=2, pady=2)

        self.dimensions_label = ttk.Label(
            frame, text="⤫", foreground="red", font="Arial 14"
        )
        self.dimensions_label.grid(row=0, column=6, padx=2, pady=2)

        ttk.Button(frame, text="Apply", command=self.app.apply_dimensions).grid(
            row=0, column=4, padx=2, pady=2
        )

    def setup_input_panels(self):
        self.setup_matrix_x_panel()
        self.setup_coefficients_b_panel()
        self.setup_noise_panel()

    def setup_matrix_x_panel(self):
        frame = ttk.LabelFrame(self.root, text="Design matrix X")
        frame.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="nsew")

        ttk.Label(frame, text="Method:").grid(row=0, column=0, padx=2, pady=2)
        self.x_choice = ttk.Combobox(
            frame, values=["Generate", "Manual", "File"], width=10, state="readonly"
        )
        self.x_choice.grid(row=0, column=1, padx=2, pady=2)
        self.x_choice.bind("<<ComboboxSelected>>", self.toggle_x_range_fields)

        self.x_range_frame = ttk.Frame(frame)
        self.x_range_frame.grid(row=1, column=0, columnspan=3, pady=2, sticky="ew")
        self.x_range_frame.grid_remove()

        ttk.Label(self.x_range_frame, text="Min:").grid(row=0, column=0, padx=2, pady=2)
        self.x_min_entry = ttk.Entry(self.x_range_frame, width=8)
        self.x_min_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(self.x_range_frame, text="Max:").grid(row=0, column=2, padx=2, pady=2)
        self.x_max_entry = ttk.Entry(self.x_range_frame, width=8)
        self.x_max_entry.grid(row=0, column=3, padx=2, pady=2)

        ttk.Label(self.x_range_frame, text="Precision:").grid(
            row=0, column=4, padx=2, pady=2
        )
        self.x_precision_entry = ttk.Entry(self.x_range_frame, width=8)
        self.x_precision_entry.grid(row=0, column=5, padx=2, pady=2)

        ttk.Button(frame, text="Apply", command=self.app.apply_X).grid(
            row=0, column=2, padx=2, pady=2
        )
        ttk.Label(
            frame,
            text=(
                f"1) File (.csv) input will automatically set the size\n"
                f"2) X values within the range of [{self.app.MIN_VAL};{self.app.MAX_VAL}]\n"
                f"3) Maximum precision - {self.app.MAX_PRECISION} decimal places"
            ),
            anchor="w",
            justify="left",
            wraplength=600
        ).grid(
            row=2,
            column=0,
            columnspan=3,
            padx=2,
            pady=(10, 2),
            sticky="w"
        )

    def setup_coefficients_b_panel(self):
        frame = ttk.LabelFrame(self.root, text="Coefficients B")
        frame.grid(row=2, column=2, columnspan=1, padx=5, pady=2, sticky="nsew")

        ttk.Label(frame, text="Bias(В₀):").grid(row=0, column=0, padx=2, pady=2)
        self.b_0_entry = ttk.Entry(frame, width=8)
        self.b_0_entry.insert(0, "1")
        self.b_0_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(frame, text="Method:").grid(row=1, column=0, padx=2, pady=2)
        self.b_choice = ttk.Combobox(
            frame, values=["Generate", "Manual"], width=10, state="readonly"
        )
        self.b_choice.grid(row=1, column=1, padx=2, pady=2)
        self.b_choice.bind("<<ComboboxSelected>>", self.toggle_b_range_fields)

        self.b_range_frame = ttk.Frame(frame)
        self.b_range_frame.grid(row=2, column=0, columnspan=3, pady=2, sticky="ew")
        self.b_range_frame.grid_remove()

        ttk.Label(self.b_range_frame, text="Min:").grid(row=0, column=0, padx=2, pady=2)
        self.b_min_entry = ttk.Entry(self.b_range_frame, width=8)
        self.b_min_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(self.b_range_frame, text="Max:").grid(row=0, column=2, padx=2, pady=2)
        self.b_max_entry = ttk.Entry(self.b_range_frame, width=8)
        self.b_max_entry.grid(row=0, column=3, padx=2, pady=2)

        ttk.Label(self.b_range_frame, text="Precision:").grid(
            row=0, column=4, padx=2, pady=2
        )
        self.b_precision_entry = ttk.Entry(self.b_range_frame, width=8)
        self.b_precision_entry.grid(row=0, column=5, padx=2, pady=2)

        ttk.Button(frame, text="Apply", command=self.app.apply_B).grid(
            row=1, column=2, padx=2, pady=2
        )
        ttk.Label(
            frame,
            text=(
                f"1) B values within the range of [{self.app.MIN_VAL};{self.app.MAX_VAL}]\n"
                f"2) Maximum precision - {self.app.MAX_PRECISION} decimal places"
            ),
            anchor="w",
            justify="left",
            wraplength=350,
        ).grid(
            row=3,
            column=0,
            columnspan=3,
            padx=2,
            pady=(10, 2),
            sticky="w"
        )

    def setup_noise_panel(self):
        frame = ttk.LabelFrame(self.root, text="Noise (ε)")
        frame.grid(row=2, column=3, padx=5, pady=2, sticky="nsew")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="E:").grid(row=0, column=0, padx=2, pady=2, sticky="e")
        self.noise_e_entry = ttk.Entry(frame, width=10)
        self.noise_e_entry.insert(0, "0")
        self.noise_e_entry.grid(row=0, column=1, padx=2, pady=2, sticky="w")

        ttk.Label(frame, text="σ:").grid(row=1, column=0, padx=2, pady=2, sticky="e")
        self.noise_sigma_entry = ttk.Entry(frame, width=10)
        self.noise_sigma_entry.insert(0, "1")
        self.noise_sigma_entry.grid(row=1, column=1, padx=2, pady=2, sticky="w")

        ttk.Button(frame, text="Apply", command=self.app.apply_noise).grid(
            row=2, column=0, columnspan=2, padx=2, pady=5
        )

        ttk.Label(
            frame,
            text=(
                f"1) E values in the range [{self.app.LOWER_LIMIT_E}, {self.app.UPPER_LIMIT_E}]\n"
                f"2) σ values in the range [{self.app.LOWER_LIMIT_SIGMA}, {self.app.UPPER_LIMIT_SIGMA}]\n"
                f"3) Precision: 9 decimal places"
            ),
            anchor="w",
            justify="left",
            wraplength=280
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            padx=2,
            pady=(10, 2),
            sticky="w"
        )

    def setup_results_panel(self):
        self.x_display = self.create_matrix_display(
            "Design matrix X", 4, 0, height=30, width=70
        )
        self.b_display = self.create_matrix_display(
            "Coefficients B", 4, 2, height=30, width=25
        )
        self.noise_display = self.create_matrix_display(
            "Noise (ε)", 4, 3, height=30, width=25
        )
        self.y_display = self.create_matrix_display(
            "Values (y)", 4, 4, height=30, width=25
        )
        self.b_hat_display = self.create_matrix_display(
            "Estimated B̂", 4, 5, height=30, width=25
        )

        metrics_frame = ttk.LabelFrame(self.root, text="Metrics (B vs B̂)")
        metrics_frame.grid(row=1, column=5, rowspan=2, padx=5, pady=2, sticky="nsew")
        self.mse_label = ttk.Label(metrics_frame, text="MSE: N/A")
        self.mse_label.pack(padx=2, pady=2)
        self.rmse_label = ttk.Label(metrics_frame, text="RMSE: N/A")
        self.rmse_label.pack(padx=2, pady=2)
        self.mae_label = ttk.Label(metrics_frame, text="MAE: N/A")
        self.mae_label.pack(padx=2, pady=2)
        self.mape_label = ttk.Label(metrics_frame, text="MAPE: N/A")
        self.mape_label.pack(padx=2, pady=2)
        ttk.Button(
            metrics_frame, text="Calculate", command=self.app.calculate_y_and_B_hat
        ).pack(anchor="se", side="bottom", padx=10, pady=10)
        ttk.Button(metrics_frame, text="Clear all", command=self.app.clear_state).pack(
            anchor="se", side="bottom", padx=10, pady=10
        )

    def create_matrix_display(
        self, title: str, row: int, col: int, height: int, width: int
    ) -> tk.Text:
        frame = ttk.LabelFrame(self.root, text=title)
        frame.grid(row=row, column=col, padx=5, pady=2, sticky="nsew")
        text = tk.Text(frame, height=height, width=width, state="disabled", wrap="none")
        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        text.pack(fill="both", expand=True)
        return text

    def toggle_x_range_fields(self, _=None):
        if self.x_choice.get() == "Generate":
            self.x_range_frame.grid()
        else:
            self.x_range_frame.grid_remove()

    def toggle_b_range_fields(self, _=None):
        if self.b_choice.get() == "Generate":
            self.b_range_frame.grid()
        else:
            self.b_range_frame.grid_remove()

    def update_display(self, text_widget: tk.Text, data: np.ndarray, precision: int):
        text_widget.config(state="normal")
        text_widget.delete(1.0, tk.END)
        for row in data:
            formatted_row = ", ".join(f"{round(val, precision)}" for val in row)
            text_widget.insert(tk.END, f"{formatted_row}\n")
        text_widget.config(state="disabled")

    def update_metrics(self, metrics: dict):
        self.mse_label.config(text=f"MSE: {metrics['mse']:.9f}")
        self.rmse_label.config(text=f"RMSE: {metrics['rmse']:.9f}")
        self.mae_label.config(text=f"MAE: {metrics['mae']:.9f}")
        self.mape_label.config(text=f"MAPE: {metrics['mape']:.9f}%")