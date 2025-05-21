import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LinearRegressionModel:

    @staticmethod
    def generate_noise(
        expected_value: float, standard_deviation: float, size: int
    ) -> np.ndarray:
        return np.random.normal(expected_value, standard_deviation, size)

    @staticmethod
    def calculate_y(
        design_matrix: np.ndarray, B: np.ndarray, bias: float, noise: np.ndarray
    ) -> np.ndarray:
        B = B.reshape(-1, 1)
        noise = noise.reshape(-1, 1)
        return np.dot(design_matrix, B) + bias + noise

    @staticmethod
    def calculate_B_hat(
        design_matrix: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        A = np.hstack([np.ones((design_matrix.shape[0], 1)), design_matrix])
        return (np.linalg.pinv(A.T @ A) @ A.T) @ Y

    @staticmethod
    def calculate_metrics(B_true: np.ndarray, B_pred: np.ndarray) -> dict:
        mse = mean_squared_error(B_true, B_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(B_true, B_pred)
        mape = np.mean(np.abs((B_true - B_pred) / B_true)) * 100
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        return metrics
