from .Loss import Loss


class MAE(Loss):
    """
    Función de pérdida Mean Absolute Error (MAE).

    Esta función calcula el promedio de las diferencias absolutas
    entre los valores predichos y los valores reales.

    Notes
    -----
    La fórmula del MAE es:

        MAE = (1 / n) * Σ |y - ŷ|
    """

    def forward(self, y_pred, y_true):
        """
        Calcula la pérdida Mean Absolute Error.

        Parameters
        ----------
        y_pred : Tensor
            Valores predichos.
        y_true : Tensor
            Valores reales esperados.

        Returns
        -------
        Tensor
            Valor del error absoluto medio.
        """
        diff = y_pred - y_true
        return diff.abs().mean()





"""
from .Loss import Loss

class MAELoss(Loss):

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return diff.abs().mean()
"""