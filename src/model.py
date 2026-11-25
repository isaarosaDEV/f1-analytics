"""Modelos de previsão de pódio e avaliação por temporada (walk-forward)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features import FEATURES, ROTULOS  # noqa: E402

MODELOS = ["logistica", "gradient_boosting"]
ORDEM = ["baseline_grid"] + MODELOS


def criar_modelo(nome):
    if nome == "logistica":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
    if nome == "gradient_boosting":
        # parâmetros conservadores: poucos dados, então evitamos decorar
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=0,
        )
    raise ValueError(f"Modelo desconhecido: {nome}")


def anos_completos(d, min_corridas=15):
    """Temporadas com corridas suficientes para servir de teste."""
    n = d.groupby("temporada")["rodada"].nunique()
    return sorted(n[n >= min_corridas].index)


def acertos_por_corrida(teste, prob):
    """Em média, quantos dos 3 pilotos de maior probabilidade subiram ao pódio."""
    t = teste[["temporada", "rodada", "podio"]].copy()
    t["prob"] = np.asarray(prob)
    por_corrida = t.groupby(["temporada", "rodada"])[["prob", "podio"]].apply(
        lambda g: g.nlargest(3, "prob")["podio"].sum()
    )
    return float(por_corrida.mean())


def avaliar_walk_forward(d, anos_teste):
    """Para cada ano de teste, treina só com anos anteriores (sem olhar o futuro)."""
    linhas = []
    for ano in anos_teste:
        treino = d[d["temporada"] < ano]
        teste = d[d["temporada"] == ano]
        if treino.empty or teste.empty or treino["podio"].nunique() < 2:
            continue

        scores = {"baseline_grid": -teste["grid_ajustado"].to_numpy(dtype=float)}
        for nome in MODELOS:
            modelo = criar_modelo(nome).fit(treino[FEATURES], treino["podio"])
            scores[nome] = modelo.predict_proba(teste[FEATURES])[:, 1]

        for nome, s in scores.items():
            linhas.append(
                {
                    "temporada": ano,
                    "modelo": nome,
                    "roc_auc": roc_auc_score(teste["podio"], s),
                    "avg_precision": average_precision_score(teste["podio"], s),
                    "acertos_por_corrida": acertos_por_corrida(teste, s),
                }
            )
    return pd.DataFrame(linhas)


def importancia_variaveis(d, ano_teste):
    """Importância por permutação no ano de teste (modelo treinado no passado)."""
    treino = d[d["temporada"] < ano_teste]
    teste = d[d["temporada"] == ano_teste]
    modelo = criar_modelo("gradient_boosting").fit(treino[FEATURES], treino["podio"])
    r = permutation_importance(
        modelo, teste[FEATURES], teste["podio"],
        scoring="roc_auc", n_repeats=10, random_state=0,
    )
    return pd.Series(r.importances_mean, index=FEATURES).sort_values()


def gerar_figuras(resultados, importancias, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    # 1) acertos de pódio por corrida: modelos x baseline, por temporada
    tab = resultados.pivot(index="temporada", columns="modelo", values="acertos_por_corrida")
    tab = tab[[c for c in ORDEM if c in tab.columns]]
    cores = {"baseline_grid": "#9a9a9a", "logistica": "#15151e", "gradient_boosting": "#e10600"}
    fig, ax = plt.subplots(figsize=(8, 5))
    largura = 0.8 / len(tab.columns)
    x = np.arange(len(tab.index))
    for i, col in enumerate(tab.columns):
        ax.bar(x + i * largura, tab[col], largura, label=col, color=cores.get(col))
    ax.set_xticks(x + largura * (len(tab.columns) - 1) / 2)
    ax.set_xticklabels(tab.index)
    ax.set_ylim(0, 3)
    ax.set_ylabel("Acertos de pódio por corrida (máx. 3)")
    ax.set_title("Quantos pilotos do pódio o modelo acerta?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "comparacao_modelos.png", dpi=150)
    plt.close(fig)

    # 2) importância das variáveis
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([ROTULOS[i] for i in importancias.index], importancias.values, color="#e10600")
    ax.set_xlabel("Queda no ROC AUC ao embaralhar a variável")
    ax.set_title("O que mais pesa na previsão do pódio")
    fig.tight_layout()
    fig.savefig(pasta / "importancia_variaveis.png", dpi=150)
    plt.close(fig)
