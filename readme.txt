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


# Ejercicios
#  Ejercicio 1 — Aprender la relación \( y = 2x \)

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

#  Ejercicio 2 — Comprender ReLU

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

| Ejercicio | Objetivo |
|---|---|
| 1 | Aprender relación lineal |
| 2 | Comprender activación ReLU |
| 3 | Aproximar función no lineal |


# GitHub del proyecto

```text
Repositorio oficial de Poliflow
```

https://github.com/Th3copolo0X/PoliFlow


# Licencia

MIT License
