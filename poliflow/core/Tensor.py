import numpy as np


def _match_shape(grad, shape):
    """
    Ajusta las dimensiones del gradiente para que coincidan con la forma objetivo.

    Esta función se utiliza principalmente durante la retropropagación
    para manejar correctamente operaciones con broadcasting.

    Parameters
    ----------
    grad : ndarray
        Arreglo de gradientes a ajustar.
    shape : tuple
        Forma objetivo del tensor.

    Returns
    -------
    ndarray
        Gradiente con dimensiones ajustadas.
    """
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad



class Tensor:
    """
    Tensor multidimensional con soporte para diferenciación automática.

    La clase Tensor almacena datos numéricos y construye dinámicamente
    un grafo computacional durante las operaciones matemáticas. Este
    grafo se utiliza posteriormente para calcular gradientes mediante
    backpropagation.

    Parameters
    ----------
    data : array-like
        Datos numéricos almacenados en el tensor.
    requieres_grad : bool, optional
        Indica si se deben calcular gradientes para este tensor.
    children : tuple, optional
        Tensores padres utilizados para construir el grafo computacional.
    op : str, optional
        Operación que creó el tensor.

    Attributes
    ----------
    data : ndarray
        Valores numéricos del tensor.
    grad : ndarray or None
        Gradientes acumulados del tensor.
    requieres_grad : bool
        Indica si el tensor participa en el cálculo de gradientes.
    """
    @property
    def shape(self):
        return self.data.shape  
    
    def __init__(self, data, requieres_grad=False, children=(), op=''):
        """
        Inicializa un objeto Tensor.
        """
        self.data = np.array(data, dtype=float)
        self.requieres_grad = requieres_grad
        self.grad = np.zeros_like(self.data) if requieres_grad else None
        self._backward = lambda: None
        self._prev = set(children)
        self.op = op
        
    def __repr__(self):

        return (
            f"Tensor(\n"
            f"  data={self.data},\n"
            f"  shape={self.shape},\n"
            f"  requieres_grad={self.requieres_grad}\n"
            f")"
        )
    def __add__(self, other):
        """
        Suma dos tensores elemento a elemento.

        Parameters
        ----------
        other : Tensor o numérico
            Tensor o valor escalar a sumar.

        Returns
        -------
        Tensor
            Tensor resultante de la suma.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='+'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += _match_shape(out.grad, self.data.shape)

            if other.requieres_grad:
                other.grad += _match_shape(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __sub__(self, other):
        """
        Resta dos tensores elemento a elemento.

        Parameters
        ----------
        other : Tensor o numérico
            Tensor o valor escalar a restar.

        Returns
        -------
        Tensor
            Tensor resultante de la resta.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data - other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='-'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += _match_shape(out.grad, self.data.shape)

            if other.requieres_grad:
                other.grad -= _match_shape(out.grad, other.data.shape)

        out._backward = _backward

        return out
    
    
    def __rsub__(self, other):
        """
        Resta inversa.

        Permite operaciones como:

            1 - tensor
        """

        other = other if isinstance(other, Tensor) else Tensor(other)

        return other - self
    

    def __mul__(self, other):
        """
        Multiplica dos tensores elemento a elemento.

        Parameters
        ----------
        other : Tensor o numérico
            Tensor o valor escalar a multiplicar.

        Returns
        -------
        Tensor
            Tensor resultante de la multiplicación.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='*'
        )

        def _backward():

            if self.requieres_grad:
                grad_self = other.data * out.grad
                self.grad += _match_shape(grad_self, self.data.shape)

            if other.requieres_grad:
                grad_other = self.data * out.grad
                other.grad += _match_shape(grad_other, other.data.shape)

        out._backward = _backward

        return out

    def __truediv__(self, other):
        """
        Divide dos tensores elemento a elemento.

        Parameters
        ----------
        other : Tensor o numérico
            Tensor o valor escalar por el cual dividir.

        Returns
        -------
        Tensor
            Tensor resultante de la división.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data / other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='/'
        )

        def _backward():

            if self.requieres_grad:
                grad_self = (1 / other.data) * out.grad
                self.grad += _match_shape(grad_self, self.data.shape)

            if other.requieres_grad:
                grad_other = (-self.data / (other.data ** 2)) * out.grad
                other.grad += _match_shape(grad_other, other.data.shape)

        out._backward = _backward

        return out

    def __pow__(self, power):
        """
        Eleva los valores del tensor a una potencia dada.

        Parameters
        ----------
        power : int o float
            Valor del exponente.

        Returns
        -------
        Tensor
            Tensor resultante de la potenciación.
        """
        out = Tensor(
            self.data ** power,
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='pow'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        """
        Niega los valores del tensor.

        Returns
        -------
        Tensor
            Tensor con valores negados.
        """
        out = Tensor(
            -self.data,
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='neg'
        )

        def _backward():

            if self.requieres_grad:
                self.grad -= out.grad

        out._backward = _backward

        return out

    def abs(self):
        """
        Calcula el valor absoluto de los elementos del tensor.

        Returns
        -------
        Tensor
            Tensor que contiene los valores absolutos.
        """
        out = Tensor(
            np.abs(self.data),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='abs'
        )

        def _backward():

            if self.requieres_grad:
                grad = np.sign(self.data)
                self.grad += grad * out.grad

        out._backward = _backward

        return out

    def sum(self):
        """
        Calcula la suma de todos los elementos del tensor.

        Returns
        -------
        Tensor
            Tensor escalar que contiene la suma.
        """
        out = Tensor(
            self.data.sum(),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='sum'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out

    def mean(self):
        """
        Calcula el promedio de todos los elementos del tensor.

        Returns
        -------
        Tensor
            Tensor escalar que contiene el promedio.
        """
        out = Tensor(
            self.data.mean(),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='mean'
        )

        def _backward():

            if self.requieres_grad:
                n = self.data.size
                self.grad += (1.0 / n) * np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        """
        Realiza multiplicación matricial entre tensores.

        Parameters
        ----------
        other : Tensor o numérico
            Tensor o matriz a multiplicar.

        Returns
        -------
        Tensor
            Tensor resultante de la multiplicación matricial.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data @ other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='@'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += out.grad @ other.data.T

            if other.requieres_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out
    
    
    def exp(self):
        """
        Calcula la exponencial elemento a elemento.

        Returns
        -------
        Tensor
            Tensor con e^x aplicado elemento a elemento.
        """

        exp_data = np.exp(self.data)

        out = Tensor(
            exp_data,
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='exp'
        )

        def _backward():

            if self.requieres_grad:

                grad_self = exp_data * out.grad
                self.grad += _match_shape(grad_self, self.data.shape)

        out._backward = _backward

        return out
    
    
    
    def log(self):
        """
        Calcula el logaritmo natural elemento a elemento.

        Returns
        -------
        Tensor
            Tensor con log(x) aplicado elemento a elemento.
        """

        epsilon = 1e-7

        out = Tensor(
            np.log(self.data + epsilon),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='log'
        )

        def _backward():

            if self.requieres_grad:

                grad_self = (1 / (self.data + epsilon)) * out.grad
                self.grad += _match_shape(grad_self, self.data.shape)

        out._backward = _backward

        return out
    
    
    

    def backward(self):
        """
        Ejecuta el algoritmo de backpropagation sobre el grafo computacional.

        Este método recorre el grafo en orden topológico inverso y
        propaga gradientes utilizando la regla de la cadena.

        Notes
        -----
        El gradiente del tensor actual se inicializa en 1 ya que
        representa la derivada del tensor respecto a sí mismo.
        """
        topo = []
        visited = set()

        def build_topo(v):

            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()






























"""
import numpy as np

def _match_shape(grad, shape):
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True) # se reduce a la dimencion correcta y mantiene la misma forma
    return grad



class Tensor:
    
    # constructor
    def __init__(self, data, requieres_grad = False, children = (), op = ''):
        self.data = np.array(data, dtype=float) # datos
        self.requieres_grad = requieres_grad # booleano que nos indica si se necesita guardar el gradiente
        self.grad = np.zeros_like(self.data) if requieres_grad else None # valor del gradiente
        self._backward = lambda: None # funcion que nos dice como retropropagar 
        self._prev = set(children) # nodos padres
        self.op = op # operacion que creó el tensor
        
       
    
    
    
    # funcion de suma
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='+'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += _match_shape(out.grad, self.data.shape)

            if other.requieres_grad:
                other.grad += _match_shape(out.grad, other.data.shape)

        out._backward = _backward

        return out
    
    
    
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data - other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='-'
        )

        def _backward():
            if self.requieres_grad:
                self.grad += _match_shape(out.grad, self.data.shape)

            if other.requieres_grad:
                other.grad -= _match_shape(out.grad, other.data.shape)

        out._backward = _backward

        return out
    
    
    
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='*'
        )

        def _backward():

            if self.requieres_grad:
                grad_self = other.data * out.grad
                self.grad += _match_shape(grad_self, self.data.shape)

            if other.requieres_grad:
                grad_other = self.data * out.grad
                other.grad += _match_shape(grad_other, other.data.shape)

        out._backward = _backward

        return out
    
    
    
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data / other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='/'
        )

        def _backward():
            if self.requieres_grad:
                grad_self = (1 / other.data) * out.grad
                self.grad += _match_shape(grad_self, self.data.shape)

            if other.requieres_grad:
                grad_other = (-self.data / (other.data ** 2)) * out.grad
                other.grad += _match_shape(grad_other, other.data.shape)

        out._backward = _backward

        return out
    
    
    
    
    def __pow__(self, power):

        out = Tensor(
            self.data ** power,
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='pow'
        )

        def _backward():
            if self.requieres_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward

        return out
    
    
    
    
    def __neg__(self):

        out = Tensor(
            -self.data,
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='neg'
        )

        def _backward():
            if self.requieres_grad:
                self.grad -= out.grad

        out._backward = _backward

        return out
    
    
    
    def abs(self):

        out = Tensor(
            np.abs(self.data),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='abs'
        )

        def _backward():

            if self.requieres_grad:
                grad = np.sign(self.data)
                self.grad += grad * out.grad

        out._backward = _backward

        return out
    
    
    
    def sum(self):

        out = Tensor(
            self.data.sum(),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='sum'
        )

        def _backward():

            if self.requieres_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out
    
    
    
    def mean(self):

        out = Tensor(
            self.data.mean(),
            requieres_grad=self.requieres_grad,
            children=(self,),
            op='mean'
        )

        def _backward():

            if self.requieres_grad:
                n = self.data.size
                        # derivada del promedio * que se aplica a todos los elementos * regla de la cadena
                self.grad += (1.0 / n) * np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out
    
    
    
    def __matmul__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)
        # tensor resultante
        out = Tensor(
            self.data @ other.data,
            requieres_grad=self.requieres_grad or other.requieres_grad,
            children=(self, other),
            op='@'
        )
        # define cómo se propaga el gradiente de esa operación específica.
        def _backward():

            if self.requieres_grad:
                self.grad += out.grad @ other.data.T

            if other.requieres_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out
    
    
    
    
    def backward(self):

        topo = [] # orden topológico
        visited = set() # nodos visitados(sin repetirse)

        def build_topo(v):
            if v not in visited: # evita repetir nodos
                visited.add(v)
                for child in v._prev: # visita nodos padres y recorre el grafo hacia atrás
                    build_topo(child)
                topo.append(v) # se agrega solo despues de visitar a los padres

        build_topo(self)

        self.grad = np.ones_like(self.data) # se inicializa el gradiente en 1

        for node in reversed(topo):
            node._backward() # se retropropagan los gradientes de cada nodo

"""
    
    
    
    