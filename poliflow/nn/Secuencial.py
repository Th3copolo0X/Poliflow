from .Modulo import Modulo


class Secuencial(Modulo):
    """
    Contenedor secuencial para capas de redes neuronales.

    Este módulo aplica una secuencia de capas en el orden en que
    fueron proporcionadas.
    """

    def __init__(self, *modules):
        """
        Inicializa el contenedor secuencial.

        Parameters
        ----------
        *modules : Module
            Secuencia de módulos de red neuronal.
        """
        super().__init__()

        self.layers = []

        for m in modules:
            self.layers.append(m)
            self.add_module(m)

    def forward(self, x):
        """
        Ejecuta una propagación hacia adelante a través de todas las capas.

        Parameters
        ----------
        x : Tensor
            Tensor de entrada.

        Returns
        -------
        Tensor
            Tensor de salida después de aplicar todas las capas.
        """
        for layer in self.layers:
            x = layer(x)

        return x




















"""
from .Module import Module


class Sequential(Module):
    
    def __init__(self, *modules): # recibe una lista de submódulos
        super().__init__() # ejecuta __init__ de Module
        
        self.layers = []
        
        for m in modules:
            self.layers.append(m)
            self.add_module(m)
    
    
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x
"""