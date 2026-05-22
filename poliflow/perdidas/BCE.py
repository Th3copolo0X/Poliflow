from .Loss import Loss


class BCE(Loss):
    """
    Función de pérdida Binary Cross Entropy (BCE).

    BCE mide la diferencia entre las probabilidades predichas
    y los valores reales en clasificación binaria.

    Notes
    -----
    La fórmula de BCE es:

        BCE = -(1 / n) * Σ [ y log(ŷ) + (1 - y) log(1 - ŷ) ]
    """

    def forward(self, y_pred, y_true):
        """
        Calcula la pérdida Binary Cross Entropy.

        Parameters
        ----------
        y_pred : Tensor
            Probabilidades predichas por el modelo.

        y_true : Tensor
            Valores reales esperados (0 o 1).

        Returns
        -------
        Tensor
            Valor de la pérdida BCE.
        """

        loss = -(
            y_true * y_pred.log() +
            (1 - y_true) * (1 - y_pred).log()
        ).mean()

        return loss