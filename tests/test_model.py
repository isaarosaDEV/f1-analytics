import numpy as np
import pandas as pd
import pytest

from src import features, model

PONTOS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _dados(n_temporadas=5, n_corridas=8, n_pilotos=20, seed=0):
    """Dados sintéticos: pilotos com habilidade fixa + ruído."""
    rng = np.random.default_rng(seed)
    habilidade = rng.normal(size=n_pilotos)
    linhas = []
    for t in range(2018, 2018 + n_temporadas):
        for r in range(1, n_corridas + 1):
            forca = habilidade + rng.normal(scale=0.5, size=n_pilotos)
            grid = np.argsort(np.argsort(-(forca + rng.normal(scale=0.7, size=n_pilotos)))) + 1
            final = np.argsort(np.argsort(-(forca + rng.normal(scale=0.8, size=n_pilotos)))) + 1
            for i in range(n_pilotos):
                linhas.append(
                    {
                        "temporada": t, "rodada": r, "piloto_id": f"p{i}",
                        "equipe_id": f"e{i // 2}", "circuito_id": f"c{r % 4}",
                        "grid": int(grid[i]), "posicao_final": int(final[i]),
                        "pontos": float(PONTOS.get(int(final[i]), 0)),
                    }
                )
    return pd.DataFrame(linhas)


def test_podio_tem_tres_por_corrida():
    d = features.adicionar_features(features.preparar_modelo(_dados()))
    assert (d.groupby(["temporada", "rodada"])["podio"].sum() == 3).all()


def test_features_nao_vazam_resultado_da_propria_corrida():
    df = _dados()
    base = features.adicionar_features(features.preparar_modelo(df))

    # altera o resultado da ÚLTIMA corrida: as features dela não podem mudar
    alterado = df.copy()
    mask = (alterado["temporada"] == 2022) & (alterado["rodada"] == 8)
    alterado.loc[mask, "posicao_final"] = 21 - alterado.loc[mask, "posicao_final"]
    alterado.loc[mask, "pontos"] = alterado.loc[mask, "posicao_final"].map(PONTOS).fillna(0)
    novo = features.adicionar_features(features.preparar_modelo(alterado))

    m = (base["temporada"] == 2022) & (base["rodada"] == 8)
    pd.testing.assert_frame_equal(
        base.loc[m, features.FEATURES].reset_index(drop=True),
        novo.loc[m, features.FEATURES].reset_index(drop=True),
    )
    # sanidade: o alvo realmente mudou
    assert not base.loc[m, "podio"].equals(novo.loc[m, "podio"])


def test_grid_dos_boxes_vira_ultimo_mais_um():
    df = _dados(n_temporadas=1, n_corridas=2)
    df.loc[0, "grid"] = 0
    d = features.adicionar_features(features.preparar_modelo(df))
    linha = d[(d["piloto_id"] == "p0") & (d["rodada"] == 1)].iloc[0]
    assert linha["grid_ajustado"] == 21


def test_avaliacao_walk_forward():
    d = features.adicionar_features(features.preparar_modelo(_dados()))
    res = model.avaliar_walk_forward(d, [2021, 2022])
    assert set(res["modelo"]) == set(model.ORDEM)
    assert res["acertos_por_corrida"].between(0, 3).all()
    assert res["roc_auc"].between(0, 1).all()
    # treino só usa o passado: o primeiro ano de teste precisa ter treino
    assert sorted(res["temporada"].unique()) == [2021, 2022]


def test_importancia_tem_todas_as_variaveis():
    d = features.adicionar_features(features.preparar_modelo(_dados()))
    imp = model.importancia_variaveis(d, 2022)
    assert set(imp.index) == set(features.FEATURES)


def test_anos_completos():
    d = features.adicionar_features(features.preparar_modelo(_dados(n_corridas=8)))
    assert model.anos_completos(d, min_corridas=5) == [2018, 2019, 2020, 2021, 2022]
    assert model.anos_completos(d, min_corridas=15) == []
