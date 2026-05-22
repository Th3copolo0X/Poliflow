import numpy as np
from .Modulo import Modulo
from ..core.Tensor import Tensor



class ReLU(Modulo):
    """
    Función de activación Rectified Linear Unit (ReLU).

    ReLU reemplaza los valores negativos por cero mientras mantiene
    los valores positivos sin cambios.

    Notes
    -----
    La función ReLU se define como:

        f(x) = max(0, x)
    """

    def forward(self, x):
        """
        Aplica la función de activación ReLU.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada.

        Returns
        -------
        Tensor
            Tensor de salida activado.
        """
        out = Tensor(
            np.maximum(0, x.data),
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='relu'
        )

        def _backward():
            if x.requieres_grad:
                grad = (x.data > 0).astype(float)
                x.grad += grad * out.grad

        out._backward = _backward

        return out



class Sigmoide(Modulo):
    """
    Función de activación Sigmoid.

    La función sigmoide transforma los valores de entrada al rango
    (0, 1), siendo útil para estimaciones de probabilidad.

    Notes
    -----
    La función sigmoide se define como:

        σ(x) = 1 / (1 + e^(-x))
    """

    def forward(self, x):
        """
        Aplica la función de activación sigmoide.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada.

        Returns
        -------
        Tensor
            Tensor de salida activado.
        """
        sig = 1 / (1 + np.exp(-x.data))

        out = Tensor(
            sig,
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='sigmoid'
        )

        def _backward():
            if x.requieres_grad:
                x.grad += (sig * (1 - sig)) * out.grad

        out._backward = _backward

        return out



class Tanh(Modulo):
    """
    Función de activación Tangente Hiperbólica (Tanh).

    La función Tanh transforma los valores de entrada al rango
    (-1, 1), centrando los datos alrededor de cero.

    Notes
    -----
    La función tangente hiperbólica se define como:

        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """

    def forward(self, x):
        """
        Aplica la función de activación Tanh.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada.

        Returns
        -------
        Tensor
            Tensor de salida activado.
        """
        t = np.tanh(x.data)

        out = Tensor(
            t,
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='tanh'
        )

        def _backward():
            if x.requieres_grad:
                x.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out



class LeakyReLU(Modulo):
    """
    Función de activación Leaky ReLU.

    Leaky ReLU permite un pequeño gradiente para valores negativos,
    evitando el problema de neuronas muertas.

    Notes
    -----
    La función Leaky ReLU se define como:

        f(x) = x       si x > 0
               αx      si x <= 0
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Aplica la función de activación Leaky ReLU.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada.

        Returns
        -------
        Tensor
            Tensor de salida activado.
        """
        out_data = np.where(x.data > 0, x.data, self.alpha * x.data)

        out = Tensor(
            out_data,
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='leaky_relu'
        )

        def _backward():
            if x.requieres_grad:
                grad = np.where(x.data > 0, 1.0, self.alpha)
                x.grad += grad * out.grad

        out._backward = _backward

        return out
    
    

class ELU(Modulo):
    """
    Función de activación Exponential Linear Unit (ELU).

    ELU suaviza los valores negativos usando una exponencial,
    ayudando a mejorar la estabilidad del entrenamiento.

    Notes
    -----
    La función ELU se define como:

        f(x) = x                     si x > 0
               α(e^x - 1)           si x <= 0
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Aplica la función de activación ELU.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada.

        Returns
        -------
        Tensor
            Tensor de salida activado.
        """
        out_data = np.where(
            x.data > 0,
            x.data,
            self.alpha * (np.exp(x.data) - 1)
        )

        out = Tensor(
            out_data,
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='elu'
        )

        def _backward():
            if x.requieres_grad:

                grad = np.where(
                    x.data > 0,
                    1.0,
                    self.alpha * np.exp(x.data)
                )

                x.grad += grad * out.grad

        out._backward = _backward

        return out





































"""
import numpy as np
from .Module import Module
from ..core.Tensor import Tensor


class ReLU(Module):
    
    def forward(self, x):
        
        # Aplicar ReLU: max(0, x)
        out = Tensor(
            np.maximum(0, x.data),
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='relu'
        )

        def _backward():
            if x.requieres_grad:
                grad = (x.data > 0).astype(float)
                x.grad += grad * out.grad

        out._backward = _backward

        return out



class Sigmoid(Module):
    
    def forward(self, x):
        
        # σ(x) = 1 / (1 + e^-x)
        sig = 1 / (1 + np.exp(-x.data))

        out = Tensor(
            sig,
            requieres_grad=x.requieres_grad,
            children=(x,),
            op='sigmoid'
        )

        def _backward():
            if x.requieres_grad:
                x.grad += (sig * (1 - sig)) * out.grad

        out._backward = _backward

        return out
"""

