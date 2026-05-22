class SGD:
    """
    Optimizador Stochastic Gradient Descent (SGD).

    Este optimizador actualiza los parámetros del modelo utilizando
    descenso por gradiente a partir de los gradientes calculados
    durante la retropropagación.

    Notes
    -----
    La regla de actualización de parámetros es:

        θ = θ - η∇J(θ)

    donde:
    - θ representa los parámetros del modelo,
    - η es la tasa de aprendizaje,
    - ∇J(θ) es el gradiente de la función de pérdida.
    """

    def __init__(self, params, lr=0.01):
        """
        Inicializa el optimizador SGD.

        Parameters
        ----------
        params : list
            Lista de tensores entrenables a optimizar.
        lr : float, optional
            Tasa de aprendizaje utilizada para actualizar
            los parámetros. El valor por defecto es 0.01.
        """
        self.params = params
        self.lr = lr

    def step(self):
        """
        Actualiza los parámetros del modelo utilizando descenso
        por gradiente.

        Cada parámetro se actualiza únicamente si requiere
        gradientes.
        """
        for p in self.params:
            if p.requieres_grad:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Reinicia los gradientes de todos los parámetros entrenables.

        Este método limpia los gradientes acumulados antes de una
        nueva pasada de backpropagation.
        """
        for p in self.params:
            if p.requieres_grad:
                p.grad = 0 * p.grad
                
                
                
                
                
                
                

"""
class SGD:

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requieres_grad:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.requieres_grad:
                p.grad = 0 * p.grad  # reinicia gradientes
                
"""