import numpy as np
from .Modulo import Modulo
from ..core.Tensor import Tensor


class Lineal(Modulo):
    """
    Capa totalmente conectada de una red neuronal.

    Esta capa aplica una transformación lineal a los datos de entrada.

    Notes
    -----
    La operación realizada es:

        y = xW + b
    """

    def __init__(self, tam_entrada, tam_salida):
        """
        Inicializa la capa lineal.

        Parameters
        ----------
        tam_entrada : int
            Número de características de entrada.
        tam_salida : int
            Número de características de salida.
        """
        super().__init__()

        W_data = np.random.randn(tam_entrada, tam_salida) * 0.2
        b_data = np.zeros((1, tam_salida))

        self.W = Tensor(W_data, requieres_grad=True)
        self.b = Tensor(b_data, requieres_grad=True)

        self.add_parameter(self.W)
        self.add_parameter(self.b)

    def forward(self, x):
        """
        Ejecuta la propagación hacia adelante de la capa lineal.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada con forma (batch_size, in_features).

        Returns
        -------
        Tensor
            Tensor de salida después de aplicar la transformación lineal.
        """
        return x @ self.W + self.b















"""
import numpy as np
from .Module import Module
from ..core.Tensor import Tensor


class Linear(Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Inicialización de pesos (pequeños valores aleatorios)
        W_data = np.random.randn(in_features, out_features) * 0.2
        b_data = np.zeros((1, out_features))
        
        # Crear tensores con gradiente
        self.W = Tensor(W_data, requires_grad=True)
        self.b = Tensor(b_data, requires_grad=True)
        
        # Registrar parámetros en el módulo
        self.add_parameter(self.W)
        self.add_parameter(self.b)
    
    
    def forward(self, x):
        # x: (batch_size, in_features)
        # W: (in_features, out_features)
        # b: (1, out_features)
        
        return x @ self.W + self.b
"""