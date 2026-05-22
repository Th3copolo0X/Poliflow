class Loss:
    """
    Clase base para funciones de pérdida.

    Las funciones de pérdida miden la diferencia entre los valores
    predichos y los valores reales durante el entrenamiento del modelo.

    Las subclases deben implementar el método forward().
    """

    def __call__(self, y_pred, y_true):
        """
        Calcula el valor de la pérdida.

        Parameters
        ----------
        y_pred : Tensor
            Valores predichos generados por el modelo.
        y_true : Tensor
            Valores reales esperados.

        Returns
        -------
        Tensor
            Valor de pérdida calculado.
        """
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        """
        Calcula la función de pérdida.

        Este método debe ser implementado por las subclases.

        Raises
        ------
        NotImplementedError
            Se lanza si la subclase no implementa el método.
        """
        raise NotImplementedError("Debe implementar forward()")



   
    