import numpy as np


class Momentum:
    """
    Optimizador SGD con Momentum.

    Este optimizador extiende Stochastic Gradient Descent (SGD)
    agregando un término de momento que acumula gradientes
    anteriores para acelerar la convergencia y reducir
    oscilaciones durante el entrenamiento.

    Notes
    -----
    Las reglas de actualización son:

        v = βv - η∇J(θ)

        θ = θ + v

    donde:
    - θ representa los parámetros del modelo,
    - η es la tasa de aprendizaje,
    - β es el factor de momentum,
    - v es la velocidad acumulada,
    - ∇J(θ) es el gradiente de la función de pérdida.
    """

    def __init__(self, params, lr=0.01, beta=0.9):
        """
        Inicializa el optimizador Momentum.

        Parameters
        ----------
        params : list
            Lista de tensores entrenables a optimizar.

        lr : float, optional
            Tasa de aprendizaje utilizada para actualizar
            los parámetros. El valor por defecto es 0.01.

        beta : float, optional
            Factor de momentum que controla cuánto de la
            velocidad previa se conserva.
            El valor por defecto es 0.9.
        """

        self.params = params
        self.lr = lr
        self.beta = beta

        # velocidades iniciales
        self.velocities = []

        for p in self.params:

            self.velocities.append(
                np.zeros_like(p.data)
            )

    def step(self):
        """
        Actualiza los parámetros utilizando SGD con Momentum.

        Cada parámetro se actualiza acumulando parte de la
        velocidad previa y el gradiente actual.
        """

        for i, p in enumerate(self.params):

            if p.requires_grad:

                # actualizar velocidad
                self.velocities[i] = (
                    self.beta * self.velocities[i]
                    - self.lr * p.grad
                )

                # actualizar parámetro
                p.data += self.velocities[i]

    def zero_grad(self):
        """
        Reinicia los gradientes de todos los parámetros entrenables.

        Este método limpia los gradientes acumulados antes de una
        nueva pasada de backpropagation.
        """

        for p in self.params:

            if p.requires_grad:

                p.grad = 0 * p.grad