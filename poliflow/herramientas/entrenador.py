from ..optim.SGD import SGD
from ..optim.Momentum import Momentum

from ..perdidas.MSE import MSE
from ..perdidas.MAE import MAE
from ..perdidas.BCE import BCE


def entrenar(
    modelo,
    x_entrenamiento,
    y_entrenamiento,
    epocas=1000,
    lr=0.025,
    perdida="mse",
    optimizador="sgd",
    beta=0.9
):
    """
    Entrena un modelo de PoliFlow.
    """

    # =========================
    # diccionario pérdidas
    # =========================

    perdidas_dict = {
        "mse": MSE,
        "mae": MAE,
        "bce": BCE
    }

    perdida = perdida.lower()

    if perdida not in perdidas_dict:
        raise ValueError(
            f"Pérdida '{perdida}' no soportada"
        )

    fn_perdida = perdidas_dict[perdida]()

    # =========================
    # diccionario optimizadores
    # =========================

    optimizadores_dict = {
        "sgd": SGD,
        "momentum": Momentum
    }

    optimizador = optimizador.lower()

    if optimizador not in optimizadores_dict:
        raise ValueError(
            f"Optimizador '{optimizador}' no soportado"
        )

    # =========================
    # crear optimizador
    # =========================

    if optimizador == "sgd":

        optimizer = SGD(
            modelo.parameters(),
            lr=lr
        )

    elif optimizador == "momentum":

        optimizer = Momentum(
            modelo.parameters(),
            lr=lr,
            beta=beta
        )

    # =========================
    # entrenamiento
    # =========================

    for epoca in range(epocas):

        # forward
        y_pred = modelo(x_entrenamiento)

        # calcular pérdida
        loss = fn_perdida(
            y_pred,
            y_entrenamiento
        )

        # backward
        optimizer.zero_grad()

        loss.backward()

        # actualizar pesos
        optimizer.step()

        # mostrar progreso
        if epoca % 100 == 0:

            print(
                f"Época {epoca}, "
                f"Pérdida: {loss.data}"
            )

    return modelo