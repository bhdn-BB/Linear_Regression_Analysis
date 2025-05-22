import contextlib
import dataclasses
import json
import os
import tkinter as tk
from tkinter import messagebox
from app_state import AppState
import numpy as np

from app_gui import AppGui
from linear_regression_model import LinearRegressionModel
from input_vectors import InputVectors


class App:
    MIN_DIMENSION_DATASET = 1
    MAX_DIMENSION_DATASET = 100_000

    MIN_VAL = 0
    MAX_VAL = 100_000_000

    LOWER_LIMIT_E = -100
    UPPER_LIMIT_E = 100

    LOWER_LIMIT_SIGMA = 0
    UPPER_LIMIT_SIGMA = 100

    MIN_PRECISION = 0
    MAX_PRECISION = 10

    def __init__(self, root: tk.Tk):
        self.input_handler = InputVectors(root)
        self.state = AppState()
        if os.path.exists("regression_state.json"):
            with contextlib.suppress(Exception):
                os.remove("regression_state.json")
                print("Cleared regression_state.json")
        self.gui = AppGui(root, self)

    def save_state(self, filename="regression_state.json"):
        with open(filename, "w") as f:
            json.dump(
                dataclasses.asdict(self.state), f, indent=4, default=lambda o: list(o)
            )

    @staticmethod
    def __check_bounds(value, lower_limit, upper_limit):
        return lower_limit <= value <= upper_limit

    def apply_dimensions(self):
        try:
            self.state.n_obs = int(self.gui.obs_entry.get())
            self.state.n_feats = int(self.gui.feat_entry.get())

            if not (
                self.__check_bounds(
                    self.state.n_obs,
                    self.MIN_DIMENSION_DATASET,
                    self.MAX_DIMENSION_DATASET,
                )
            ) or not (
                self.__check_bounds(
                    self.state.n_feats,
                    self.MIN_DIMENSION_DATASET,
                    self.MAX_DIMENSION_DATASET,
                )
            ):
                raise ValueError(
                    f"Dimensions must be between {self.MIN_DIMENSION_DATASET} and {self.MAX_DIMENSION_DATASET}"
                )
            self.gui.dimensions_label.config({"text": "✓", "foreground": "green"})
            self.save_state()
            return
        except ValueError as e:
            self.gui.obs_entry.delete(0, tk.END)
            self.gui.feat_entry.delete(0, tk.END)
            self.gui.dimensions_label.config({"text": "⤫", "foreground": "red"})
            messagebox.showerror("Error", f"Invalid dimensions: {e}")
            return

    def apply_X(self):

        if not self.gui.x_choice.get():
            messagebox.showerror("Error", "Select input method for X")
            return

        match self.gui.x_choice.get():
            case "Generate":
                try:
                    if not self.state.n_obs or not self.state.n_feats:
                        messagebox.showerror("Error", "Apply X dimensions first")
                        return
                    min_bounds_x_program = float(self.gui.x_min_entry.get())
                    max_bounds_x_program = float(self.gui.x_max_entry.get())
                    self.state.x_precision = int(self.gui.x_precision_entry.get())

                    if not (
                        self.__check_bounds(
                            min_bounds_x_program, self.MIN_VAL, self.MAX_VAL
                        )
                    ) or not (
                        self.__check_bounds(
                            max_bounds_x_program, self.MIN_VAL, self.MAX_VAL
                        )
                    ):
                        self.gui.x_min_entry.delete(0, tk.END)
                        self.gui.x_max_entry.delete(0, tk.END)
                        self.gui.x_precision_entry.delete(0, tk.END)
                        messagebox.showerror(
                            "Error",
                            f"Bounds X must be in the range [{self.MIN_VAL}, {self.MAX_VAL}]",
                        )
                        return

                    if min_bounds_x_program >= max_bounds_x_program:
                        self.gui.x_min_entry.delete(0, tk.END)
                        self.gui.x_max_entry.delete(0, tk.END)
                        self.gui.x_precision_entry.delete(0, tk.END)
                        messagebox.showerror("Error", "Min must be less than Max")
                        return
                    if not (
                        self.__check_bounds(
                            self.state.x_precision,
                            self.MIN_PRECISION,
                            self.MAX_PRECISION,
                        )
                    ):
                        self.gui.x_precision_entry.delete(0, tk.END)
                        self.gui.x_min_entry.delete(0, tk.END)
                        self.gui.x_max_entry.delete(0, tk.END)
                        messagebox.showerror(
                            "Error",
                            f"Precision must be between {self.MIN_PRECISION} and {self.MAX_PRECISION}",
                        )
                        return
                    self.state.data_X = InputVectors.generate_random_matrix(
                        min_bounds_x_program,
                        max_bounds_x_program,
                        self.state.n_obs,
                        self.state.n_feats,
                        self.state.x_precision,
                    )
                except ValueError as e:
                    self.gui.x_min_entry.delete(0, tk.END)
                    self.gui.x_max_entry.delete(0, tk.END)
                    self.gui.x_precision_entry.delete(0, tk.END)
                    messagebox.showerror("Error", f"Invalid range/precision: {e}")

            case "Manual":
                if not self.state.n_obs or not self.state.n_feats:
                    messagebox.showerror("Error", "Apply X dimensions first")
                    return
                self.state.data_X = self.input_handler.input_matrix_gui(
                    "Enter Feature Matrix X", self.state.n_obs, self.state.n_feats
                )

            case "File":
                if (
                    data_X := InputVectors.load_from_file()
                ) is not None and np.all(data_X):
                    self.state.data_X = data_X
                    self.state.n_obs, self.state.n_feats = data_X.shape
                    self.gui.obs_entry.delete(0, tk.END)
                    self.gui.obs_entry.insert(0, data_X.shape[0])
                    self.gui.feat_entry.delete(0, tk.END)
                    self.gui.feat_entry.insert(0, data_X.shape[1])
                    self.gui.dimensions_label.config({"text": "✓", "foreground": "green"})
                    self.save_state()
            case _:
                pass

        if np.all(self.state.data_X):
            self.gui.update_display(
                self.gui.x_display, self.state.data_X, self.state.x_precision
            )
            self.save_state()
        else:
            messagebox.showerror("Error", "Failed to load Feature Matrix X")

    def clear_state(self):
        self.state = AppState()
        self.gui.update_display(self.gui.x_display, np.array([]), 0)
        self.gui.update_display(self.gui.y_display, np.array([]), 0)
        self.gui.update_display(self.gui.b_display, np.array([]), 0)
        self.gui.update_display(self.gui.b_hat_display, np.array([]), 0)
        self.gui.update_display(self.gui.noise_display, np.array([]), 0)
        self.gui.dimensions_label.config({"text": "⤫", "foreground": "red"})
        self.gui.feat_entry.delete(0, tk.END)
        self.gui.obs_entry.delete(0, tk.END)
        self.gui.x_min_entry.delete(0, tk.END)
        self.gui.x_max_entry.delete(0, tk.END)
        self.gui.x_precision_entry.delete(0, tk.END)
        self.gui.b_min_entry.delete(0, tk.END)
        self.gui.b_max_entry.delete(0, tk.END)
        self.gui.b_precision_entry.delete(0, tk.END)
        self.gui.b_0_entry.delete(0, tk.END)
        self.gui.b_0_entry.insert(0, "1")
        self.gui.noise_e_entry.delete(0, tk.END)
        self.gui.noise_e_entry.insert(0, "0")
        self.gui.noise_sigma_entry.delete(0, tk.END)
        self.gui.noise_sigma_entry.insert(0, "1")
        self.gui.update_metrics({"mape": 0, "mse": 0, "rmse": 0, "mae": 0})

    def apply_noise(self):
        if not self.state.n_obs or not self.state.n_feats:
            messagebox.showerror("Error", "Apply X dimensions first")
            return
        try:
            noise_e = float(self.gui.noise_e_entry.get())
            noise_sigma = float(self.gui.noise_sigma_entry.get())

            if not (
                self.__check_bounds(noise_e, self.LOWER_LIMIT_E, self.UPPER_LIMIT_E)
            ):
                self.gui.noise_e_entry.delete(0, tk.END)
                self.gui.noise_e_entry.insert(0, "0")
                messagebox.showerror(
                    "Error",
                    f"E must be in the range [{self.LOWER_LIMIT_E}, {self.UPPER_LIMIT_E}]",
                )
                return
            if not (
                self.__check_bounds(
                    noise_sigma, self.LOWER_LIMIT_SIGMA, self.UPPER_LIMIT_SIGMA
                )
            ):
                self.gui.noise_sigma_entry.delete(0, tk.END)
                self.gui.noise_sigma_entry.insert(0, "1")
                messagebox.showerror(
                    "Error",
                    f"σ must be in the range [{self.LOWER_LIMIT_SIGMA}, {self.UPPER_LIMIT_SIGMA}]",
                )
                return

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid range/σ, E: {e}")
            return

        self.state.noise = LinearRegressionModel.generate_noise(
            noise_e, noise_sigma, self.state.n_obs, self.MAX_PRECISION
        )
        self.gui.update_display(
            self.gui.noise_display,
            np.array([[n] for n in self.state.noise]),
            self.MAX_PRECISION,
        )
        self.save_state()

    def apply_B(self):
        if not self.state.n_obs or not self.state.n_feats:
            messagebox.showerror("Error", "Apply X dimensions first")
            return
        if not self.gui.b_choice.get():
            messagebox.showerror("Error", "Select input method for B")
            return
        try:
            self.state.b_0 = float(self.gui.b_0_entry.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid B_0: {e}")
            return

        match self.gui.b_choice.get():
            case "Generate":
                try:
                    min_bounds_b_program = float(self.gui.b_min_entry.get())
                    max_bounds_b_program = float(self.gui.b_max_entry.get())
                    self.state.b_precision = int(self.gui.b_precision_entry.get())

                    if not (
                        self.__check_bounds(
                            min_bounds_b_program, self.MIN_VAL, self.MAX_VAL
                        )
                    ) or not (
                        self.__check_bounds(
                            max_bounds_b_program, self.MIN_VAL, self.MAX_VAL
                        )
                    ):
                        self.gui.b_min_entry.delete(0, tk.END)
                        self.gui.b_max_entry.delete(0, tk.END)
                        self.gui.b_precision_entry.delete(0, tk.END)
                        messagebox.showerror(
                            "Error",
                            f"Bounds B must be in the range [{self.MIN_VAL}, {self.MAX_VAL}]",
                        )
                        return

                    if min_bounds_b_program >= max_bounds_b_program:
                        self.gui.b_min_entry.delete(0, tk.END)
                        self.gui.b_max_entry.delete(0, tk.END)
                        self.gui.b_precision_entry.delete(0, tk.END)
                        messagebox.showerror("Error", "Min must be less than Max")
                        return
                    if not (
                        self.__check_bounds(
                            self.state.b_precision,
                            self.MIN_PRECISION,
                            self.MAX_PRECISION,
                        )
                    ):
                        self.gui.b_precision_entry.delete(0, tk.END)
                        self.gui.b_min_entry.delete(0, tk.END)
                        self.gui.b_max_entry.delete(0, tk.END)
                        messagebox.showerror(
                            "Error",
                            f"Precision must be between {self.MIN_PRECISION} and {self.MAX_PRECISION}",
                        )
                        return
                    self.state.data_B = InputVectors.generate_random_matrix(
                        min_bounds_b_program,
                        max_bounds_b_program,
                        self.state.n_feats,
                        1,
                        self.state.b_precision,
                    )
                except ValueError as e:
                    self.gui.b_precision_entry.delete(0, tk.END)
                    self.gui.b_min_entry.delete(0, tk.END)
                    self.gui.b_max_entry.delete(0, tk.END)
                    messagebox.showerror("Error", f"Invalid range/precision: {e}")
                    return
            case "Manual":
                self.state.data_B = self.input_handler.input_matrix_gui(
                    "Enter Coefficients B", self.state.n_feats, 1
                )
            case _:
                pass
        target_B = np.insert(self.state.data_B, 0, self.state.b_0, axis=0)
        self.gui.update_display(
            self.gui.b_display, target_B, self.state.b_precision
        )
        self.save_state()

    def calculate_y_and_B_hat(self):
        X_np = self.state.data_X
        B_np = self.state.data_B
        noise_np = self.state.noise

        self.state.data_Y = LinearRegressionModel.calculate_y(
            X_np, B_np, self.state.b_0, noise_np
        )
        if self.state.data_Y is None:
            messagebox.showerror("Error", "Failed to calculate y")
            return
        self.gui.update_display(
            self.gui.y_display,
            self.state.data_Y,
            self.state.b_precision,
        )

        Y_np = self.state.data_Y
        try:
            self.state.B_hat = LinearRegressionModel.calculate_B_hat(
                X_np, Y_np
            )
            np.round(self.state.B_hat, self.state.b_precision)
        except Exception as e:
            messagebox.showerror("Calculate B̂ Error", f"Error calculating B̂: {e}")

        if self.state.B_hat is None:
            messagebox.showerror("Error", "Failed to calculate B̂")
            return

        self.gui.update_display(
            self.gui.b_hat_display,
            self.state.B_hat,
            self.MAX_PRECISION,
        )

        B_pred = self.state.B_hat
        metrics = LinearRegressionModel.calculate_metrics(B_np, B_pred[1:])
        self.gui.update_metrics(metrics)
        self.save_state()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
