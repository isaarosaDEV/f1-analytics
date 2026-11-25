"""Variáveis para prever o pódio usando só informação disponível ANTES da largada.

Regra de ouro (evita vazamento de dados): toda variável de "forma" usa
shift(1), ou seja, só enxerga corridas anteriores à corrida analisada.
"""
import pandas as pd

FEATURES = [
    "grid_ajustado",
    "forma_piloto",
    "forma_equipe",
    "media_pontos_temporada",
    "media_pos_circuito",
]

ROTULOS = {
    "grid_ajustado": "Posição de largada",
    "forma_piloto": "Forma do piloto (últimas 5 corridas)",
    "forma_equipe": "Forma da equipe (últimas 5 corridas)",
    "media_pontos_temporada": "Média de pontos na temporada",
    "media_pos_circuito": "Posição média no circuito",
}


def preparar_modelo(df):
    """Remove linhas sem dados. Largadas dos boxes (grid = 0) são mantidas."""
    d = df.dropna(subset=["grid", "posicao_final"]).copy()
    d["grid"] = d["grid"].astype(int)
    d["posicao_final"] = d["posicao_final"].astype(int)
    d["pontos"] = d["pontos"].fillna(0.0)
    return d


def _media_anterior(serie, janela=None):
    """Média só do passado: a corrida atual nunca entra na própria variável."""
    anterior = serie.shift(1)
    if janela:
        return anterior.rolling(janela, min_periods=1).mean()
    return anterior.expanding().mean()


def adicionar_features(df, janela=5):
    d = df.sort_values(["temporada", "rodada", "piloto_id"]).reset_index(drop=True)
    d["corrida_idx"] = d.groupby(["temporada", "rodada"]).ngroup()  # ordem cronológica

    # largada dos boxes (grid = 0) vira "último + 1"
    grid_max = d.groupby("corrida_idx")["grid"].transform("max")
    d["grid_ajustado"] = d["grid"].where(d["grid"] > 0, grid_max + 1)

    d["forma_piloto"] = d.groupby("piloto_id")["pontos"].transform(
        lambda s: _media_anterior(s, janela)
    )
    d["media_pontos_temporada"] = d.groupby(["temporada", "piloto_id"])[
        "pontos"
    ].transform(_media_anterior)
    d["media_pos_circuito"] = d.groupby(["piloto_id", "circuito_id"])[
        "posicao_final"
    ].transform(_media_anterior)

    # forma da equipe: média de pontos por carro, só de corridas anteriores
    equipe = (
        d.groupby(["equipe_id", "corrida_idx"], as_index=False)["pontos"]
        .mean()
        .sort_values(["equipe_id", "corrida_idx"])
    )
    equipe["forma_equipe"] = equipe.groupby("equipe_id")["pontos"].transform(
        lambda s: _media_anterior(s, janela)
    )
    d = d.merge(
        equipe[["equipe_id", "corrida_idx", "forma_equipe"]],
        on=["equipe_id", "corrida_idx"],
        how="left",
    )

    d["podio"] = (d["posicao_final"] <= 3).astype(int)
    return d
