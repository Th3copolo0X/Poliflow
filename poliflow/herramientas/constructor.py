from ..nn.Secuencial import Secuencial
from ..nn.Lineal import Lineal

from ..nn.Activacion import (
    ReLU,
    Sigmoide,
    Tanh,
    LeakyReLU,
    ELU
)


def construir_red_densa(
    tamaño_entrada,
    tamaño_salida,
    num_capas_ocultas=2,
    tamaño_capa_oculta=16,
    fn_activacion="relu",
    tarea="regresion"
):
    """
    Construye automáticamente una red neuronal densa rectangular.
    """

    capas = []

    # =========================
    # diccionario activaciones
    # =========================

    activacion_dict = {
        "relu": ReLU,
        "sigmoide": Sigmoide,
        "tanh": Tanh,
        "leakyrelu": LeakyReLU,
        "elu": ELU
    }

    fn_activacion = fn_activacion.lower()

    if fn_activacion not in activacion_dict:
        raise ValueError(
            f"Activación '{fn_activacion}' no soportada"
        )

    clase_activacion = activacion_dict[fn_activacion]

    # =========================
    # primera capa
    # =========================

    capas.append(
        Lineal(tamaño_entrada, tamaño_capa_oculta)
    )

    capas.append(
        clase_activacion()
    )

    # =========================
    # capas ocultas
    # =========================

    for _ in range(num_capas_ocultas - 1):

        capas.append(
            Lineal(tamaño_capa_oculta, tamaño_capa_oculta)
        )

        capas.append(
            clase_activacion()
        )

    # =========================
    # capa de salida
    # =========================

    capas.append(
        Lineal(tamaño_capa_oculta, tamaño_salida)
    )

    # clasificación binaria
    if tarea == "binaria":
        capas.append(Sigmoide())

    return Secuencial(*capas)