import numpy as np
import pandas as pd

from src import analysis, load, transform

CORRIDAS = [
    {
        "season": "2023",
        "round": "1",
        "raceName": "Bahrain Grand Prix",
        "date": "2023-03-05",
        "Circuit": {
            "circuitId": "bahrain",
            "circuitName": "Bahrain International Circuit",
            "Location": {"locality": "Sakhir", "country": "Bahrain"},
        },
        "Results": [
            {
                "position": "1", "positionText": "1", "points": "25", "grid": "1",
                "laps": "57", "status": "Finished",
                "Driver": {"driverId": "max_verstappen", "givenName": "Max", "familyName": "Verstappen"},
                "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
            },
            {
                "position": "20", "positionText": "R", "points": "0", "grid": "0",
                "laps": "3", "status": "Engine",
                "Driver": {"driverId": "piloto_x", "givenName": "Piloto", "familyName": "X"},
                "Constructor": {"constructorId": "equipe_x", "name": "Equipe X"},
            },
        ],
    }
]


def test_achatar_resultados():
    df = transform.achatar_resultados(CORRIDAS)
    assert len(df) == 2
    primeiro = df.iloc[0]
    assert primeiro["piloto"] == "Max Verstappen"
    assert primeiro["grid"] == 1 and primeiro["posicao_final"] == 1
    assert primeiro["pontos"] == 25.0


def test_carga_idempotente():
    conn = load.conectar(":memory:")
    df = transform.achatar_resultados(CORRIDAS)
    load.carregar(conn, df)
    load.carregar(conn, df)  # rodar de novo não pode duplicar
    assert len(load.ler_resultados(conn)) == 2


def _dados_sinteticos():
    rng = np.random.default_rng(0)
    linhas = []
    for temporada in (2021, 2022):
        for rodada in range(1, 6):
            chegada = rng.permutation(20) + 1
            for grid in range(1, 21):
                linhas.append(
                    {"temporada": temporada, "rodada": rodada, "grid": grid,
                     "posicao_final": int(chegada[grid - 1])}
                )
    return pd.DataFrame(linhas)


def test_preparar_remove_largada_dos_boxes():
    df = transform.achatar_resultados(CORRIDAS)
    d = analysis.preparar(df)
    assert len(d) == 1  # a linha com grid = 0 sai


def test_analises_basicas():
    d = analysis.preparar(_dados_sinteticos())
    assert analysis.taxa_vitoria_por_grid(d).between(0, 100).all()
    assert set(analysis.correlacao_por_temporada(d).index) == {2021, 2022}
    m = analysis.matriz_grid_chegada(d)
    assert m.shape == (20, 20)
    assert abs(m.sum(axis=1) - 100).max() < 1e-6
