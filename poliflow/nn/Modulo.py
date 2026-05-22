class Modulo:
    """
    Clase base para todos los módulos de redes neuronales.

    Un módulo puede contener parámetros entrenables y otros submódulos,
    permitiendo la construcción de arquitecturas complejas de redes
    neuronales.

    Las subclases deben implementar el método forward().
    """

    def __init__(self):
        """
        Inicializa el módulo.
        """
        self._parameters = []
        self._modules = []

    def parameters(self):
        """
        Devuelve todos los parámetros entrenables del módulo.

        Este método recopila recursivamente los parámetros del módulo
        actual y de todos sus submódulos.

        Returns
        -------
        list
            Lista de tensores entrenables.
        """
        params = []

        for p in self._parameters:
            params.append(p)

        for m in self._modules:
            params.extend(m.parameters())

        return params

    def add_parameter(self, param):
        """
        Registra un parámetro entrenable en el módulo.

        Parameters
        ----------
        param : Tensor
            Tensor entrenable a registrar.
        """
        self._parameters.append(param)

    def add_module(self, module):
        """
        Registra un submódulo dentro del módulo.

        Parameters
        ----------
        module : Module
            Submódulo de red neuronal.
        """
        self._modules.append(module)

    def zero_grad(self):
        
        #Reinicia los gradientes de todos los parámetros entrenables.
        
        for p in self.parameters():
            if p.requires_grad:
                p.grad = 0 * p.grad


    def __call__(self, *args):
        """
        Ejecuta la propagación hacia adelante del módulo.

        Returns
        -------
        Tensor
            Tensor de salida producido por el módulo.
        """
        return self.forward(*args)

    def forward(self, *args, **kwargs):
        """
        Define el cálculo forward del módulo.

        Este método debe ser implementado por las subclases.

        Raises
        ------
        NotImplementedError
            Se lanza si la subclase no implementa el método.
        """
        raise NotImplementedError("Debe implementar forward()")










"""
class Module:
    
    # constructor
    def __init__(self):
        self._parameters = [] # tensores entrenables
        self._modules = [] #subcapas(otros modulos)
    
    
    def parameters(self):
        params = [] # lista final de todos los parámetros del modelo
        
        # parámetros propios
        for p in self._parameters:
            params.append(p)
        
        # parámetros de submódulos
        for m in self._modules:
            params.extend(m.parameters()) # recursivo
        
        return params
    
    
    def add_parameter(self, param): # agrega tensor como entrenable
        self._parameters.append(param)
    
    
    def add_module(self, module): # agrega submódulos
        self._modules.append(module)
    
    
    def zero_grad(self): # reinicia gradientes(funcion que esta en el optimizador)
        for p in self.parameters():
            if p.requires_grad:
                p.grad = 0 * p.grad
    
    
    def __call__(self, *args): # para poder llamar a model()
        return self.forward(*args) # ejecuta el metodo forward
    
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Debe implementar forward()")
"""