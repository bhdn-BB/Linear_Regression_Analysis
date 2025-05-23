import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LinearRegressionModel:

    # <summary>
    # Генерує шум із заданим математичним сподіванням і стандартним відхиленням.
    # </summary>
    # <param name="expected_value">Математичне сподівання нормального розподілу</param>
    # <param name="standard_deviation">Стандартне відхилення нормального розподілу</param>
    # <param name="size">Кількість значень шуму</param>
    # <param name="precision">Кількість знаків після коми для округлення</param>
    # <returns>Масив значень шуму</returns>
    @staticmethod
    def generate_noise(
        expected_value: float, standard_deviation: float, size: int, precision: int
    ) -> np.ndarray:
        return np.round(np.random.normal(expected_value, standard_deviation, size), precision)

    # <summary>
    # Обчислює вектор значень Y, використовуючи вхідний датасет, вектор коефіцієнтів знучущості, біас і шум.
    # </summary>
    # <param name="design_matrix">Матриця спостережень</param>
    # <param name="B">Вектор істинних коефіцієнтів</param>
    # <param name="bias">Значення зсуву</param>
    # <param name="noise">Масив шуму</param>
    # <returns>Розраховані значення Y</returns>
    @staticmethod
    def calculate_y(
        design_matrix: np.ndarray, B: np.ndarray, bias: float, noise: np.ndarray
    ) -> np.ndarray:
        B = B.reshape(-1, 1)
        noise = noise.reshape(-1, 1)
        return np.dot(design_matrix, B) + bias + noise

    # <summary>
    # Обчислює оцінку вектора коефіцієнтів B за методом найменших квадратів.
    # </summary>
    # <param name="design_matrix">Матриця спостережень</param>
    # <param name="Y">Вектор відповідей</param>
    # <returns>Оцінений вектор коефіцієнтів B_hat</returns>
    @staticmethod
    def calculate_B_hat(
        design_matrix: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        A = np.hstack([np.ones((design_matrix.shape[0], 1)), design_matrix])
        return (np.linalg.pinv(A.T @ A) @ A.T) @ Y

    # <summary>
    # Обчислює метрики якості оцінки коефіцієнтів: MSE, RMSE, MAE, MAPE.
    # </summary>
    # <param name="B_true">Істинний вектор коефіцієнтів</param>
    # <param name="B_pred">Оцінений вектор коефіцієнтів</param>
    # <returns>Словник з метриками якості</returns>
    @staticmethod
    def calculate_metrics(B_true: np.ndarray, B_pred: np.ndarray) -> dict:
        mse = mean_squared_error(B_true, B_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(B_true, B_pred)
        mape = np.mean(np.abs((B_true - B_pred) / B_true)) * 100
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        return metrics
