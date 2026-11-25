"""Primeira análise: quanto a posição de largada decide o resultado da corrida?"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera imagens sem precisar de janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def preparar(df):
    """Remove linhas sem dados e largadas dos boxes (grid = 0)."""
    d = df.dropna(subset=["grid", "posicao_final"]).copy()
    d = d[d["grid"] > 0]
    d["grid"] = d["grid"].astype(int)
    d["posicao_final"] = d["posicao_final"].astype(int)
    d["venceu"] = d["posicao_final"] == 1
    d["podio"] = d["posicao_final"] <= 3
    return d


def taxa_vitoria_por_grid(d, max_grid=10):
    """% de vitórias para cada posição de largada (1 a max_grid)."""
    sub = d[d["grid"] <= max_grid]
    return sub.groupby("grid")["venceu"].mean().mul(100)


def vitoria_pole_por_temporada(d):
    return d[d["grid"] == 1].groupby("temporada")["venceu"].mean().mul(100)


def correlacao_por_temporada(d):
    """Correlação de Spearman entre largada e chegada, por temporada."""
    return d.groupby("temporada")[["grid", "posicao_final"]].apply(
        lambda g: g["grid"].corr(g["posicao_final"], method="spearman")
    )


def matriz_grid_chegada(d, n=20):
    """Para cada largada, % das vezes que o piloto terminou em cada posição."""
    sub = d[(d["grid"] <= n) & (d["posicao_final"] <= n)]
    tabela = pd.crosstab(sub["grid"], sub["posicao_final"], normalize="index") * 100
    return tabela.reindex(index=range(1, n + 1), columns=range(1, n + 1), fill_value=0)


def resumo(d):
    pole = d[d["grid"] == 1]
    return {
        "temporadas": f"{d['temporada'].min()}-{d['temporada'].max()}",
        "corridas": d.groupby(["temporada", "rodada"]).ngroups,
        "vitoria_pole_pct": round(pole["venceu"].mean() * 100, 1),
        "podio_pole_pct": round(pole["podio"].mean() * 100, 1),
        "correlacao_media": round(correlacao_por_temporada(d).mean(), 2),
    }


def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def gerar_figuras(d, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    # 1) taxa de vitória por posição de largada
    taxa = taxa_vitoria_por_grid(d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(taxa.index, taxa.values, color="#e10600")
    ax.set_xticks(taxa.index)
    ax.set_xlabel("Posição de largada")
    ax.set_ylabel("Vitórias (%)")
    ax.set_title("Chance de vitória por posição de largada")
    _salvar(fig, pasta / "vitoria_por_grid.png")

    # 2) correlação largada x chegada ao longo das temporadas
    corr = correlacao_por_temporada(d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(corr.index, corr.values, marker="o", color="#15151e")
    ax.set_xlabel("Temporada")
    ax.set_ylabel("Correlação de Spearman")
    ax.set_title("O quanto a largada explica a chegada, por temporada")
    ax.grid(alpha=0.3)
    _salvar(fig, pasta / "correlacao_por_temporada.png")

    # 3) mapa de calor largada x chegada
    m = matriz_grid_chegada(d)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m.values, cmap="magma_r", origin="upper")
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels(m.columns, fontsize=7)
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels(m.index, fontsize=7)
    ax.set_xlabel("Posição final")
    ax.set_ylabel("Posição de largada")
    ax.set_title("Onde cada posição de largada termina (%)")
    fig.colorbar(im, ax=ax, label="% das largadas")
    _salvar(fig, pasta / "mapa_largada_chegada.png")
