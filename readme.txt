# Poliflow

Poliflow es un framework de Deep Learning desarrollado en Python desde cero, diseñado con fines educativos y de investigación.  
Incluye tensores, capas neuronales, funciones de activación, optimizadores y entrenamiento de modelos.

---

# Instalación

```bash
pip install poliflow
```

---

# Características

- Tensores personalizados
- Redes neuronales 
- Capas  Lineal
- Funciones de activación
- Función de pérdida 
- Optimizador 
- Entrenamiento de modelo
- Predicción de datos

# Ejemplo de un Tensor

Los `Tensor` son la base de Poliflow.  
Permiten realizar operaciones matemáticas y calcular gradientes automáticamente mediante *autograd*.

---

## Características principales

- Soporte para operaciones matemáticas
- Backpropagation automático
- Compatible con NumPy
- Grafo computacional dinámico
- Operadores intuitivos (`+`, `-`, `*`, `@`, `/`)
- Soporte para gradientes

---

## Crear un Tensor

```python
from poliflow.core.Tensor import Tensor

x = Tensor([1, 2, 3])

print(x)
```

### Salida

```python
Tensor(
  data=[1. 2. 3.],
  shape=(3,),
  requieres_grad=False
)
```

### Explicación

Aquí se crea un tensor unidimensional a partir de una lista de Python.

---

## Tensor con gradientes

Para entrenar redes neuronales normalmente necesitamos calcular derivadas.

```python
from poliflow.core.Tensor import Tensor

x = Tensor([2.0], requieres_grad=True)

print(x)
```

### Explicación

El parámetro:

```python
requiere_grad=True
```

indica que Poliflow debe almacenar operaciones para calcular gradientes posteriormente.

---

## Operaciones matemáticas

Los tensores soportan operaciones básicas.

```python
from poliflow.core.Tensor import Tensor

a = Tensor([2.0], requieres_grad=True)
b = Tensor([3.0], requieres_grad=True)

c = a + b
d = a * b

print(c.data)
print(d.data)
```

### Explicación

Poliflow construye automáticamente el grafo computacional de las operaciones.

---

## Multiplicación matricial

También puedes trabajar con matrices.

```python
from poliflow.core.Tensor import Tensor

x = Tensor([[1, 2]])
w = Tensor([[3], [4]])

y = x @ w

print(y.data)
```

### Salida

```python
Tensor([[11.]])
```

### Explicación

El operador:

```python
@
```

realiza multiplicación matricial.

---

## Backpropagation automático

Poliflow puede calcular derivadas automáticamente.

```python
from poliflow.core.Tensor import Tensor

x = Tensor([2.0], requieres_grad=True)

y = x * x
y.backward()

print(x.grad)
```

### Salida

```python
[4.]
```

### Explicación

La función:

<img src="https://latex.codecogs.com/svg.image?y=x^2">

tiene derivada:

<img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}(x^2)=2x">

Cuando:

```python
x = 2
```

el gradiente es:

```python
4
```

---

## Operaciones soportadas

Actualmente Poliflow soporta:

```python
+
-
*
/
@
pow()
sum()
mean()
exp()
log()
abs()
```

---

## Ejemplo completo

```python
from poliflow import Tensor

# Datos de entrada
x = Tensor([2.0], requiere_grad=True)

# Peso
w = Tensor([3.0], requiere_grad=True)

# Bias
b = Tensor([1.0], requiere_grad=True)

# Forward
y = x * w + b

# Backward
y.backward()

print("Salida:", y.data)
print("Gradiente de x:", x.grad)
print("Gradiente de w:", w.grad)
print("Gradiente de b:", b.grad)
```

### Explicación

Poliflow construye automáticamente un grafo computacional dinámico
a partir de las operaciones entre tensores.

La operación realizada es:

```math
y = xw + b
```

Sustituyendo valores:

```math
y = (2)(3) + 1
```

```math
y = 6 + 1
```

```math
y = 7
```

Durante el `forward`, Poliflow guarda todas las operaciones realizadas:

```text
x ----\
       (*) ---- (+) ---> y
w ----/          ^
                 |
                 b
```

Después, al ejecutar:

```python
y.backward()
```

Poliflow aplica automáticamente backpropagation y calcula las derivadas
de cada variable usando la regla de la cadena.

## Gradiente respecto a `x`

<img src="https://latex.codecogs.com/svg.image?\frac{\partial%20y}{\partial%20x}=w">

<img src="https://latex.codecogs.com/svg.image?\frac{\partial%20y}{\partial%20x}=3">

## Gradiente respecto a `w`

<img src="https://latex.codecogs.com/svg.image?\frac{\partial%20y}{\partial%20w}=x">

<img src="https://latex.codecogs.com/svg.image?\frac{\partial%20y}{\partial%20w}=2">

## Gradiente respecto a `b`

<img src="https://latex.codecogs.com/svg.image?\frac{\partial%20y}{\partial%20b}=1">

Por lo tanto:

```python
Salida: [7.]
Gradiente de x: [3.]
Gradiente de w: [2.]
Gradiente de b: [1.]
```

Esto permite entrenar modelos automáticamente mediante descenso por gradiente,
igual que en frameworks de deep learning modernos.



# Módulo `nn` en Poliflow

El módulo `nn` (neural networks) de Poliflow proporciona las piezas fundamentales para construir redes neuronales de forma modular.

Su objetivo es abstraer operaciones complejas en capas reutilizables.

---

#  Estructura general del módulo `nn`

Dentro de `nn` normalmente encontramos:

- `Lineal` → capa totalmente conectada 
- `Secuencial` → contenedor de capas en secuencia
- `Modulo`→
- (Opcional) funciones de activación como ReLU, Sigmoid, Tanh

---

#  Capa Linear (Dense Layer)

La capa `Linear` es la base de las redes neuronales.

## ¿Qué hace?

Realiza la transformación:

<img src="https://latex.codecogs.com/svg.image?y=xW+b">

donde:

- `x` = entrada
- `W` = pesos
- `b` = sesgo (bias)

---

## Ejemplo de uso

```python
from poliflow.nn.Lineal import Lineal
from poliflow.core.Tensor import Tensor

capa = Linear(tam_entrada=3, tam_salida=2)

x = Tensor([[1.0, 2.0, 3.0]])

y = capa(x)

print(y)
```

---

## Internamente

La capa `Linear`:

- Inicializa pesos aleatorios `W`
- Inicializa bias `b`
- Aplica multiplicación matricial
- Usa autograd para permitir backpropagation

---

## ¿Qué representa la salida?

La variable y es la salida de la capa Lineal, es decir, el resultado de aplicar una transformación lineal a la entrada:

<img src="https://latex.codecogs.com/svg.image?y=xW+b">

## Ejemplo de salida
Tensor(
  data=[[0.03735598 1.67213206]],
  shape=(1, 2),
  requieres_grad=True
)

## Esto significa:

y tiene forma (1, 2)
Hay 1 muestra (batch size = 1)
La capa tiene 2 neuronas de salida
Cada valor corresponde a una neurona distinta

---

#  Múltiples capas (stack manual)
Las redes neuronales pueden construirse **apilando capas lineales una encima de otra**, donde la salida de una capa se convierte en la entrada de la siguiente.

Esto se conoce como un **stack manual de capas**.

---

##  Idea principal

Una red de múltiples capas realiza transformaciones sucesivas:

<img src="https://latex.codecogs.com/svg.image?x\rightarrow%20h=f_1(x)\rightarrow%20y=f_2(h)">

Donde:

- \( x \): entrada
- \( h \): capa oculta
- \( y \): salida final
- \( f_1, f_2 \): capas lineales

---

##  Arquitectura del ejemplo

- Capa 1: `Lineal(4 → 3)`
- Capa 2: `Lineal(3 → 1)`

Esto significa:

- Entrada de 4 características → salida de 3
- Luego de 3 → salida final de 1

---

Puedes apilar capas manualmente:

```python
from poliflow.nn.Lineal import Lineal
from poliflow.core.Tensor import Tensor

c1 = Lineal(4, 3)
c2 = Lineal(3, 1)

x = Tensor([[1.0, 2.0, 3.0, 4.0]])

h = c1(x)
salida = c2(h)

print(salida)
```

---
Fórmulas matemáticas

Cada capa lineal realiza la operación:

<img src="https://latex.codecogs.com/svg.image?y=xW+b">

Entonces:

### Primera capa

<img src="https://latex.codecogs.com/svg.image?h=xW_1+b_1">

---

### Segunda capa

<img src="https://latex.codecogs.com/svg.image?y=hW_2+b_2">

---

### Composición completa

<img src="https://latex.codecogs.com/svg.image?y=(xW_1+b_1)W_2+b_2">

---

## Interpretación

- La red aprende **representaciones intermedias** en `h`
- Cada capa transforma los datos a otro espacio de características
- La segunda capa trabaja sobre una representación más abstracta

---






#  Secuencial (modelo en cadena)

`Secuencial` permite encadenar capas de forma automática.

## ¿Qué hace?

Ejecuta las capas en orden:

```text
entrada → capa1 → capa2 → capa3 → salida
```

---

## Ejemplo básico

```python
from poliflow.nn.Secuencial import Secuencial
from poliflow.nn.Lineal import Lineal
from poliflow.core.Tensor import Tensor

modelo = Secuencial(
    Lineal(4, 8),
    Lineal(8, 3),
    Lineal(3, 1)
)

x = Tensor([[1.0, 2.0, 3.0, 4.0]])

y = modelo(x)

print(y)
```

---

## Cómo funciona internamente

`Secuencial`:

1. Recibe una lista de capas
2. Guarda el orden
3. En el paso_adelante:
   - pasa la salida de una capa como entrada a la siguiente
4. Devuelve el resultado final

---

## Flujo de ejecución

```text
x
 ↓
Lineal(4 → 8)
 ↓
Lineal(8 → 3)
 ↓
Lineal(3 → 1)
 ↓
salida
```

---
## Representación matemática

Un modelo `Secuencial` es una **composición de funciones**:

<img src="https://latex.codecogs.com/svg.image?f(x)=f_3(f_2(f_1(x)))">

Donde cada \( f_i \) es una capa lineal:

<img src="https://latex.codecogs.com/svg.image?f_i(x)=xW_i+b_i">

---

## Expansión completa

Sustituyendo cada capa:

<img src="https://latex.codecogs.com/svg.image?f(x)=((xW_1+b_1)W_2+b_2)W_3+b_3">

---

## Interpretación

- Cada capa aplica una transformación lineal
- Los parámetros \( W_i \) y \( b_i \) se aprenden durante el entrenamiento
- El modelo completo es una composición encadenada de transformaciones


# Diferencia: Linear vs Sequential

| Componente   | Función |
|-------------|--------|
| Lineal      | Una sola transformación lineal |
| Secuencial  | Encadena múltiples capas |

---

#  Retropropagación (Backpropagation)

La **retropropagación** es el algoritmo que permite calcular automáticamente los **gradientes** de cada parámetro en la red neuronal usando la regla de la cadena.

Esto es lo que hace posible el aprendizaje en redes neuronales.

---

##  Idea principal

Después del *forward pass*, se calcula el gradiente desde la salida hacia atrás:

<img src="https://latex.codecogs.com/svg.image?\text{forward:}\;x\rightarrow y">

<img src="https://latex.codecogs.com/svg.image?\text{backward:}\;\frac{\partial L}{\partial y}\rightarrow\frac{\partial L}{\partial x}">

---

## Ejemplo en Poliflow

```python
from poliflow.nn.Secuencial import Secuencial
from poliflow.nn.Lineal import Lineal
from poliflow.core.Tensor import Tensor

model = Secuencial(
    Lineal(2, 4),
    Lineal(4, 1)
)

x = Tensor([[0.5, 1.5]])

# Forward pass
y = model(x)

# Backpropagation
y.backward()

print("Salida:", y.data)



```
## Flujo 
Backward pass (gradientes)

```text

 ↓
Linear(2 → 4)
 ↓
h
 ↓
Linear(4 → 1)
 ↓
y

Retropropagacíon (gradientes)
dy/dy = 1
   ↓
∂L/∂h
   ↓
Linear(4 → 1)ᵀ
   ↓
∂L/∂x
   ↓
Linear(2 → 4)ᵀ
```
---
##  Representación matemática

---

## Forward

```text
h = xW₁ + b₁

y = hW₂ + b₂

Backpropagation (regla de la cadena):

∂L/∂W₂ = hᵀ · (∂L/∂y)
∂L/∂W₁ = xᵀ · (∂L/∂h)
∂L/∂x = (∂L/∂h) · W₁ᵀ
```


## Diagrama del proceso

![Backpropagation GIF](https://www.birow.com/storage/news/2024/10/backpropagation/backpropagation.gif)

**Fuente:** Birow — https://www.birow.com/news/backpropagation-explained


#  Funciones de Activación en Poliflow

Las funciones de activación introducen **no linealidad** en la red neuronal y permiten que el modelo aprenda patrones complejos.

En Poliflow, cada activación está implementada como un `Modulo` con soporte completo de **autograd (forward + backward)**.

---

# 1. ReLU (Rectified Linear Unit)

## Definición

<img src="https://latex.codecogs.com/svg.image?f(x)=\max(0,x)">

---

## Derivada (backprop)

\[
\frac{d}{dx} f(x) =
\begin{cases}
1 & x > 0 \\
0 & x \le 0
\end{cases}
\]

---

## Comportamiento

```text
x < 0  →  0
x ≥ 0  →  x
```

## Uso
```python
from poliflow.nn.Activacion import ReLU
from poliflow.core.Tensor import Tensor

# Crear la activación
act = ReLU()

# Entrada
x = Tensor([[1.0, -2.0, 3.0]])

# Aplicar ReLU
y = act(x)

print("Salida:", y)
```

## salida
```text
Salida: Tensor(
  data=[[1. 0. 3.]],)
```
# 2. Sigmoid

## Definición

<img src="https://latex.codecogs.com/svg.image?f(x)=\frac{1}{1+e^{-x}}">

---

## Derivada (backprop)

<img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}f(x)=f(x)(1-f(x))">

---

## Comportamiento

```text
x → -∞   → 0
x → 0    → 0.5
x → +∞   → 1
```
## Uso
```python
from poliflow.nn.Activacion import Sigmoide
from poliflow.core.Tensor import Tensor

# Crear activación
act = Sigmoide()

# Entrada
x = Tensor([[1.0, -2.0, 3.0]])

# Aplicar Sigmoid
y = act(x)

print("Salida:", y)
```

## salida
```text
Salida: Tensor(
  data=[[0.7310586  0.11920292 0.95257413]],)
```

---

# 3. Tanh

## Definición

<img src="https://latex.codecogs.com/svg.image?f(x)=\tanh(x)">

---

## Derivada (backprop)

<img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}f(x)=1-\tanh^2(x)">
---

## Comportamiento

```text
x → -∞   → -1
x → 0    → 0
x → +∞   → 1
```

## Uso

```python
from poliflow.nn.Activacion import Tanh
from poliflow.core.Tensor import Tensor

# Crear activación
act = Tanh()

# Entrada
x = Tensor([[1.0, -2.0, 3.0]])

# Aplicar Tanh
y = act(x)

print("Salida:", y)
```

## Salida

```text
Salida: Tensor(
  data=[[ 0.76159416 -0.96402758  0.99505475]],)
```

---

# 4. Leaky ReLU

## Definición

<img src="https://latex.codecogs.com/svg.image?f(x)=\begin{cases}x&x>0\\\alpha%20x&x\le0\end{cases}">

---

## Derivada (backprop)

<img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}f(x)=\begin{cases}1&x>0\\\alpha&x\le0\end{cases}">

---

## Comportamiento

```text
x > 0   → x
x ≤ 0   → αx
```

## Uso

```python
from poliflow.nn.Activacion import LeakyReLU
from poliflow.core.Tensor import Tensor

# Crear activación
act = LeakyReLU(alpha=0.01)

# Entrada
x = Tensor([[1.0, -2.0, 3.0]])

# Aplicar Leaky ReLU
y = act(x)

print("Salida:", y)
```

## Salida

```text
Salida: Tensor(
  data=[[ 1.   -0.02  3.  ]],)
```

---


# Herramienta

Poliflow incluye herramientas que permiten:

- construir redes neuronales rápidamente
- entrenar modelos con pocas líneas
- visualizar arquitecturas automáticamente

Esto evita escribir manualmente cada capa y el ciclo de entrenamiento.

---

# Construcción automática de redes densas

La función `construir_red_densa` permite crear una red neuronal completa de forma automática.

---

## Ejemplo de uso

```python
from poliflow.core.Tensor import Tensor
from poliflow.herramientas.constructor import construir_red_densa
from poliflow.herramientas.entrenador import entrenar

# Datos simples
X = Tensor([[1.0], [2.0], [3.0], [4.0]])
y = Tensor([[2.0], [4.0], [6.0], [8.0]])

# Construcción automática de la red
modelo = construir_red_densa(
    tamaño_entrada=1,
    tamaño_salida=1,
    num_capas_ocultas=2,
    tamaño_capa_oculta=8,
    fn_activacion="relu",
    tarea="regresion"
)

# Entrenamiento automático

entrenar(
    modelo=modelo,
    x_entrenamiento=X,
    y_entrenamiento=y,
    epocas=100,
    lr=0.01,
    perdida="mse"
)

# Predicción
pred = modelo(Tensor([[5.0]]))

print(pred)
```

---

# ¿Qué hace `construir_red_densa`?

Esta herramienta:

- crea automáticamente capas lineales
- agrega funciones de activación
- construye una arquitectura secuencial completa
- permite configurar profundidad y tamaño de la red

---

# Arquitectura generada

El ejemplo anterior crea una red similar a:

```text
1 → 8 → 8 → 1
```

Donde:

- `1` = tamaño de entrada
- `8` = neuronas ocultas
- `1` = salida

---

# ¿Qué hace `entrenar`?

La función `entrenar` automatiza:

- forward propagation
- cálculo de pérdida
- backpropagation
- actualización de parámetros
- ciclo completo de entrenamiento

---

# Internamente

Durante el entrenamiento Poliflow ejecuta:

```python
y_pred = modelo(X)

loss = perdida(y_pred, y)

loss.backward()

optimizador.step()
```

Todo esto ocurre automáticamente.

---

# Ventajas

Las herramientas de alto nivel permiten:

- crear modelos rápidamente
- reducir código repetitivo
- experimentar arquitecturas fácilmente
- entrenar redes neuronales con pocas líneas

---

# Intuición

En lugar de construir manualmente cada capa:

```python
Lineal(...)
ReLU()
Lineal(...)
ReLU()
Lineal(...)
```

Poliflow puede generar toda la arquitectura automáticamente usando únicamente parámetros de configuración.

----

# Visualización de redes neuronales

Poliflow permite visualizar arquitecturas de redes neuronales automáticamente utilizando `NetworkX` y `Matplotlib`.

La herramienta `dibujar_red` genera un grafo donde:

- cada nodo representa una neurona
- cada línea representa una conexión
- las capas se organizan automáticamente

---

## Ejemplo de uso

```python
from poliflow.herramientas.constructor import construir_red_densa
from poliflow.visual.Grafo import dibujar_red

modelo = construir_red_densa(
    tamaño_entrada=3,
    tamaño_salida=1,
    num_capas_ocultas=2,
    tamaño_capa_oculta=4,
    fn_activacion="relu",
    tarea="regresion"
)

dibujar_red(modelo)
```

---

## ¿Qué se visualizará?

La herramienta mostrará:

- neuronas de entrada
- capas ocultas
- neuronas de salida
- conexiones entre capas

Además:

- entrada → color verde
- capas ocultas → color azul
- salida → color rojo

---

## Resultado esperado

Inserta aquí una imagen generada de la arquitectura:

```markdown
<img src=(https://raw.githubusercontent.com/Th3copolo0X/Poliflow/main/grafo_computacional.png)>
```


# Ejemplos de implementacion para regresión y clasificacion binaria
#  Ejemplo 1 — Aprender la relación \( y = 2x \)

##  Objetivo

Entrenar una red neuronal simple para aprender la relación lineal:

```text
y = 2x
```

---

##  Datos

```python
X = [[1], [2], [3], [4], [5]]
Y = [[2], [4], [6], [8], [10]]
```

---

##  Arquitectura

```python
Linear(1,1)
```

La red tiene:

- 1 entrada
- 1 salida
- Sin capas ocultas

---

##  Parámetros

```python
learning_rate = 0.01
epochs = 1000
```

---

##  Código completo

```python
from poliflow.nn.Lineal import Lineal
from poliflow.core.Tensor import Tensor

# Modelo
model = Lineal(1, 1)

# Datos
X = Tensor([[1], [2], [3], [4], [5]])
Y = Tensor([[2], [4], [6], [8], [10]])

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):

    # Forward
    pred = model(X)

    # Loss MSE
    loss = ((pred - Y) * (pred - Y)).mean()

    # Backward
    loss.backward()

    # Actualización manual
    for param in model.parameters():
        param.data -= learning_rate * param.grad

    # Limpiar gradientes
    model.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data}")

print("Predicción final:")
print(model(X))
```

---

##  Resultado esperado

La red aprenderá pesos cercanos a:

```text
w ≈ 2
b ≈ 0
```

---

#  Ejemplo 2 — Comprender ReLU

##  Objetivo

Comprender cómo funciona la activación ReLU eliminando valores negativos.

---

##  Datos

```python
X = [[-3], [-2], [-1], [0], [1], [2], [3]]

Y = [[0], [0], [0], [0], [1], [2], [3]]
```

---

##  Arquitectura

```python
Sequential(
    Linear(1,1),
    ReLU()
)
```

---

##  Parámetros

```python
learning_rate = 0.01
epochs = 1500
```

---

##  Código completo

```python
from poliflow.nn.Secuencial import Secuencial
from poliflow.nn.Lineal import Lineal
from poliflow.nn.Activacion import ReLU
from poliflow.core.Tensor import Tensor

# Modelo
model = Secuencial(
    Lineal(1,1),
    ReLU()
)

# Datos
X = Tensor([[-3], [-2], [-1], [0], [1], [2], [3]])
Y = Tensor([[0], [0], [0], [0], [1], [2], [3]])

learning_rate = 0.01
epochs = 400

for epoch in range(epochs):

    # Forward
    pred = model(X)

    # Loss
    loss = ((pred - Y) * (pred - Y)).mean()

    # Backward
    loss.backward()

    # Update
    for param in model.parameters():
        param.data -= learning_rate * param.grad

    model.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data}")

print("Predicciones:")
print(model(X))
```

---

##  Qué observar

```text
x < 0  →  salida ≈ 0
x > 0  →  salida ≈ x
```

La red aprenderá el comportamiento típico de ReLU.

---



##  Qué observar

- La red aprende una relación no lineal
- La capa oculta permite representar curvas
- ReLU introduce no linealidad
- Más neuronas = mayor capacidad de aproximación

---

#  Resumen

| Ejemplo | Objetivo |
|---|---|
| 1 | Aprender relación lineal |
| 2 | Comprender activación ReLU |
| 3 | Aproximar función no lineal |

---


# Ejemplo 3 — Regresión con red  neuronal personalizada

## Objetivo

Regresión del precio de casas utilizando PoliFlow.

Este ejemplo implementa una red neuronal construida
manualmente para resolver un problema de regresión utilizando
el dataset House Price Regression.

La red neuronal se construye capa por capa utilizando:
- capas Lineal
- activación Tanh
- pérdida MSE
- optimizador SGD

Además, se visualiza la arquitectura de la red al finalizar.

---

## Dataset

```text
House Price Regression Dataset
```

Variables utilizadas:

- Square_Footage
- Num_Bedrooms
- Num_Bathrooms
- Year_Built
- Lot_Size
- Garage_Size
- Neighborhood_Quality

Variable objetivo:

```text
House_Price
```

---

## Código completo

```python
import pandas as pd

from poliflow.core.Tensor import Tensor
from poliflow.nn.Lineal import Lineal
from poliflow.nn.Secuencial import Secuencial
from poliflow.nn.Activacion import Tanh
from poliflow.perdidas.MSE import MSE
from poliflow.optim.SGD import SGD
from poliflow.visual.Grafo import dibujar_red

# ==================================================
# cargar dataset
# ==================================================
df = pd.read_csv("data/house_price_regression_dataset.csv")

# ==================================================
# separar features y target
# ==================================================
#
# X:
# variables de entrada
#
# y:
# precio de la casa
#
# ==================================================
X = df.drop(columns=["House_Price"]).values
y = df["House_Price"].values.reshape(-1, 1)

# ==================================================
# normalizar datos
# ==================================================
#
# Se utiliza estandarización:
#
# x = (x - media) / desviación
#
# ==================================================
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()

# ==================================================
# dividir train / test
# ==================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==================================================
# convertir a tensores
# ==================================================
X_train = Tensor(X_train)
y_train = Tensor(y_train)

X_test = Tensor(X_test)
y_test = Tensor(y_test)


# ==================================================
# construir modelo manualmente
# ==================================================
#
# Arquitectura:
#
# 7 -> 12 -> 16 -> 25 -> 20 -> 16 -> 8 -> 1
#
# ==================================================
model = Secuencial(
    Lineal(7, 12),
    Tanh(),
    Lineal(12, 16),
    Tanh(),
    Lineal(16, 25),
    Tanh(),
    Lineal(25, 20),
    Tanh(),
    Lineal(20, 16),
    Tanh(),
    Lineal(16, 8),
    Tanh(),
    Lineal(8, 1)
)

# ==================================================
# función de pérdida
# ==================================================
loss_fn = MSE()

# ==================================================
# optimizador
# ==================================================
optimizer = SGD(model.parameters(), lr=0.025)

# ==================================================
# entrenamiento
# ==================================================
epochs = 1000

for epoch in range(epochs):
    # ------------------------------
    # forward
    # ------------------------------
    y_pred = model(X_train)

    # ------------------------------
    # calcular pérdida
    # ------------------------------
    loss = loss_fn(y_pred, y_train)

    # ------------------------------
    # limpiar gradientes
    # ------------------------------
    optimizer.zero_grad()

    # ------------------------------
    # backward
    # ------------------------------
    loss.backward()

    # ------------------------------
    # actualizar pesos
    # ------------------------------
    optimizer.step()

    # ------------------------------
    # mostrar progreso
    # ------------------------------
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, " f"Train Loss: {loss.data}")

# ==================================================
# evaluación
# ==================================================
y_pred_test = model(X_test)
test_loss = loss_fn(y_pred_test, y_test)
print("\nTest Loss:", test_loss.data)

# ==================================================
# visualizar arquitectura
# ==================================================
dibujar_red(model)
```


---

## Qué observar
- Se utiliza la librería pandas para la preparacion de los datos
- La pérdida disminuye gradualmente
- El modelo aprende relaciones no lineales
- Las activaciones afectan el comportamiento de la red
- Redes más profundas pueden aproximar funciones complejas

---

# Ejemplo 4 — Regresión usando Constructor

## Objetivo

Construir automáticamente una red neuronal densa rectangular
utilizando las herramientas de PoliFlow.

Este ejemplo demuestra cómo generar arquitecturas completas
con una sola función.

---

## Dataset

```text
House Price Regression Dataset
```

---

## Arquitectura generada

```python
import pandas as pd

from poliflow.core.Tensor import Tensor
from poliflow.herramientas.Constructor import construir_red_densa
from poliflow.herramientas.Entrenador import entrenar
from poliflow.visual.Grafo import dibujar_red
from poliflow.perdidas.MSE import MSE

# ==================================================
# cargar dataset
# ==================================================
df = pd.read_csv("data/house_price_regression_dataset.csv")

# ==================================================
# separar features y target
# ==================================================
X = df.drop(columns=["House_Price"]).values
y = df["House_Price"].values.reshape(-1, 1)

# ==================================================
# normalizar datos
# ==================================================
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()

# ==================================================
# dividir train / test
# ==================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==================================================
# convertir a tensores
# ==================================================
X_train = Tensor(X_train)
y_train = Tensor(y_train)

X_test = Tensor(X_test)
y_test = Tensor(y_test)

# ==================================================
# construir modelo automáticamente
# ==================================================
#
# Arquitectura generada:
#
# 7 -> 20 -> 20 -> 20 -> 20 -> 20 -> 20 -> 1
#
# con activación Tanh entre capas
#
# ==================================================
model = construir_red_densa(
    tamaño_entrada=7,
    tamaño_salida=1,
    num_capas_ocultas=6,
    tamaño_capa_oculta=20,
    fn_activacion="tanh",
    tarea="regresion"
)

# ==================================================
# entrenamiento automático
# ==================================================
entrenar(
    modelo=model,
    x_entrenamiento=X_train,
    y_entrenamiento=y_train,
    epocas=1000,
    lr=0.025,
    perdida="mse",
    optimizador="sgd"
)

# ==================================================
# evaluación
# ==================================================
loss_fn = MSE()
y_pred_test = model(X_test)
test_loss = loss_fn(y_pred_test, y_test)
print("\nTest Loss:", test_loss.data)

# ==================================================
# visualizar arquitectura
# ==================================================
dibujar_red(model)
```

---

## Qué aprenderás

- Uso del constructor automático
- Generación dinámica de arquitecturas
- Redes rectangulares
- Automatización de entrenamiento
- Entrenamiento simplificado

---

## Qué observar

- El constructor genera automáticamente:
  - capas ocultas
  - activaciones
  - capa de salida

- La red puede entrenarse sin construir manualmente
  cada capa

- Diferentes activaciones producen comportamientos distintos

---

# Ejemplo 5 — Clasificación binaria con red neuronal personalizada

## Objetivo

Clasificación binaria del dataset Breast Cancer Wisconsin
utilizando PoliFlow.

Este ejemplo construye manualmente una red neuronal densa
sin utilizar las herramientas automáticas de construcción
y entrenamiento.

El objetivo es clasificar tumores como:
- 0 -> benigno
- 1 -> maligno
---

## Código completo

```python
import pandas as pd

from poliflow.core.Tensor import Tensor
from poliflow.nn.Lineal import Lineal
from poliflow.nn.Secuencial import Secuencial
from poliflow.nn.Activacion import LeakyReLU
from poliflow.nn.Activacion import Sigmoide
from poliflow.perdidas.BCE import BCE
from poliflow.optim.Momentum import Momentum
from poliflow.visual.Grafo import dibujar_red

# ==================================================
# cargar dataset
# ==================================================
df = pd.read_csv("data/breast_cancer_clasification.csv")

# ==================================================
# eliminar columna ID
# ==================================================
df = df.drop(columns=["id"])

# ==================================================
# convertir etiquetas
# M -> maligno -> 1
# B -> benigno -> 0
# ==================================================
df["diagnosis"] = df["diagnosis"].map({"M": 1,"B": 0})

# ==================================================
# separar features y target
# ==================================================
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values.reshape(-1, 1)

# ==================================================
# normalizar features
# ==================================================
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ==================================================
# dividir train / test
# ==================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==================================================
# convertir a tensores
# ==================================================
X_train = Tensor(X_train)
y_train = Tensor(y_train)

X_test = Tensor(X_test)
y_test = Tensor(y_test)

# ==================================================
# construir modelo manualmente
# ==================================================
#
# Arquitectura:
#
# 30 -> 34 -> 36 -> 38 -> 35 -> 36 -> 1
#
# con activación LeakyReLU entre capas
# y Sigmoide al final para clasificación binaria
#
# ==================================================
model = Secuencial(
    Lineal(30, 34),
    LeakyReLU(),
    Lineal(34, 36),
    LeakyReLU(),
    Lineal(36, 38),
    LeakyReLU(),
    Lineal(38, 35),
    LeakyReLU(),
    Lineal(35, 33),
    LeakyReLU(),
    Lineal(33, 1),
    Sigmoide()
)

# ==================================================
# función de pérdida
# ==================================================
loss_fn = BCE()

# ==================================================
# optimizador
# ==================================================
optimizer = Momentum(model.parameters(), lr=0.001, beta=0.9)

# ==================================================
# entrenamiento
# ==================================================
epochs = 4000

for epoch in range(epochs):
    # ------------------------------
    # forward
    # ------------------------------
    y_pred = model(X_train)

    # ------------------------------
    # calcular pérdida
    # ------------------------------
    loss = loss_fn(y_pred, y_train)

    # ------------------------------
    # limpiar gradientes
    # ------------------------------
    optimizer.zero_grad()

    # ------------------------------
    # backward
    # ------------------------------
    loss.backward()

    # ------------------------------
    # actualizar parámetros
    # ------------------------------
    optimizer.step()

    # ------------------------------
    # mostrar progreso
    # ------------------------------
    if epoch % 100 == 0:
        print(f"Época {epoch}, "f"Pérdida: {loss.data}")

# ==================================================
# evaluación
# ==================================================
y_pred_test = model(X_test)

# ==================================================
# convertir probabilidades
# a clases binarias
# ==================================================
predicciones = (y_pred_test.data > 0.5).astype(int)

# ==================================================
# accuracy
# ==================================================
accuracy = (predicciones == y_test.data).mean()
print("\nAccuracy:", accuracy)

# ==================================================
# mostrar primeras predicciones
# ==================================================
print("\nPrimeras predicciones:\n")

for i in range(10):
    probabilidad = y_pred_test.data[i][0]
    prediccion = predicciones[i][0]
    valor_real = int(y_test.data[i][0])
    print(f"Probabilidad: {probabilidad:.4f} | " f"Predicción: {prediccion} | " f"Real: {valor_real}")

# ==================================================
# visualizar arquitectura
# ==================================================
dibujar_red(model)
```

---


## Qué observar

- La salida final representa probabilidades
- Valores cercanos a:
  - 0 → clase negativa
  - 1 → clase positiva

- BCE penaliza predicciones incorrectas
- Sigmoide transforma la salida al rango (0,1)

---

# Ejemplo 6 — Clasificación binaria usando Constructor

## Objetivo

Crear automáticamente una red neuronal para clasificación binaria
utilizando las herramientas de construcción y entrenamiento
de PoliFlow.

---

## Código completo

```python
import pandas as pd

from poliflow.core.Tensor import Tensor
from poliflow.herramientas.Constructor import construir_red_densa
from poliflow.herramientas.Entrenador import entrenar
from poliflow.visual.Grafo import dibujar_red

# ==================================================
# cargar dataset
# ==================================================
df = pd.read_csv("data/breast_cancer_clasification.csv")

# ==================================================
# eliminar columna ID
# ==================================================
df = df.drop(columns=["id"])

# ==================================================
# convertir etiquetas
#
# M -> maligno -> 1
# B -> benigno -> 0
# ==================================================
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# ==================================================
# separar features y target
# ==================================================
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values.reshape(-1, 1)

# ==================================================
# normalizar features
#
# Se utiliza normalización Z-score:
#
# x_normalizado = (x - media) / desviación
# ==================================================
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ==================================================
# dividir dataset
#
# 80% entrenamiento
# 20% prueba
# ==================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==================================================
# convertir a tensores
# ==================================================
X_train = Tensor(X_train)
y_train = Tensor(y_train)

X_test = Tensor(X_test)
y_test = Tensor(y_test)


# ==================================================
# construir red neuronal automáticamente
# ==================================================
#
# Arquitectura generada:
#
# 30 -> 36 -> 36 -> 36 -> 36 -> 36 -> 1
#
# con:
# - LeakyReLU entre capas
# - Sigmoide al final
#
# debido a que:
#
# tarea = "binaria"
#
# ==================================================
model = construir_red_densa(
    tamaño_entrada=30,
    tamaño_salida=1,
    num_capas_ocultas=5,
    tamaño_capa_oculta=36,
    fn_activacion="leakyrelu",
    tarea="binaria"
)

# ==================================================
# entrenar modelo
# ==================================================
#
# Se utiliza:
#
# - BCE como función de pérdida
# - Momentum como optimizador
#
# ==================================================
entrenar(
    modelo=model,
    x_entrenamiento=X_train,
    y_entrenamiento=y_train,
    epocas=4000,
    lr=0.001,
    perdida="bce",
    optimizador="momentum",
    beta=0.9
)

# ==================================================
# realizar predicciones
# ==================================================
y_pred = model(X_test)

# ==================================================
# convertir probabilidades
# a clases binarias
#
# Si probabilidad > 0.5:
#     clase = 1
# Si no:
#     clase = 0
# ==================================================
predicciones = (y_pred.data > 0.5).astype(int)

# ==================================================
# calcular accuracy
# ==================================================
accuracy = (predicciones == y_test.data).mean()
print("\nAccuracy:", accuracy)

# ==================================================
# mostrar primeras predicciones
#==================================================

print("\nPrimeras predicciones:\n")

for i in range(10):
    probabilidad = y_pred.data[i][0]
    prediccion = predicciones[i][0]
    valor_real = int(y_test.data[i][0])
    print(f"Probabilidad: {probabilidad:.4f} | " f"Predicción: {prediccion} | " f"Real: {valor_real}")

# ==================================================
# visualizar arquitectura de la red
# ==================================================
dibujar_red(model)

```

---

## Qué aprenderás

- Construcción automática de modelos
- Clasificación binaria simplificada
- Uso automático de Sigmoide
- Entrenamiento mediante herramientas de PoliFlow
- Configuración rápida de redes neuronales

---

## Qué observar

- El constructor agrega automáticamente:
  - capas ocultas
  - activaciones
  - capa de salida
  - Sigmoide para clasificación binaria

- El entrenamiento puede realizarse con pocas líneas de código

- PoliFlow permite construir modelos completos
  sin definir manualmente cada capa

---

## Resumen de capacidades actuales de PoliFlow

Actualmente PoliFlow soporta:

- Tensores con autograd
- Backpropagation automático
- Redes neuronales densas
- Arquitecturas secuenciales
- Funciones de activación:
  - ReLU
  - Sigmoide
  - Tanh
  - LeakyReLU
  - ELU

- Funciones de pérdida:
  - MSE
  - MAE
  - BCE

- Optimizadores:
  - SGD
  - Momentum

- Problemas de:
  - regresión
  - clasificación binaria

- Construcción automática de arquitecturas
- Entrenamiento simplificado
- Visualización de redes neuronales

---


# GitHub del proyecto

```text
Repositorio oficial de Poliflow
```

https://github.com/Th3copolo0X/PoliFlow


# Licencia

MIT License
