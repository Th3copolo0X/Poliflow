from .Loss import Loss


class MSE(Loss):
    """
    Función de pérdida Mean Squared Error (MSE).

    Esta función calcula el promedio de las diferencias cuadradas
    entre los valores predichos y los valores reales.

    Notes
    -----
    La fórmula del MSE es:

        MSE = (1 / n) * Σ (y - ŷ)^2
    """

    def forward(self, y_pred, y_true):
        """
        Calcula la pérdida Mean Squared Error.

        Parameters
        ----------
        y_pred : Tensor
            Valores predichos.
        y_true : Tensor
            Valores reales esperados.

        Returns
        -------
        Tensor
            Valor del error cuadrático medio.
        """
        diff = y_pred - y_true
        return (diff * diff).mean()




"""
from .Loss import Loss

class MSELoss(Loss):

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return (diff * diff).mean()
"""
