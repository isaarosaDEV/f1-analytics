"""Dashboard de F1: largada x chegada, pilotos e desempenho do modelo.

Rodar a partir da raiz do projeto:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import analysis, load  # noqa: E402

METRICAS = RAIZ / "reports" / "metricas_por_temporada.csv"
FIG_IMPORTANCIA = RAIZ / "reports" / "figures" / "importancia_variaveis.png"
VERMELHO = "#e10600"
CORES_MODELO = {
    "baseline_grid": "#9a9a9a",
    "logistica": "#15151e",
    "gradient_boosting": VERMELHO,
}

st.set_page_config(page_title="F1 Analytics", page_icon="🏎️", layout="wide")


@st.cache_data
def carregar(mtime):
    """Lê o banco; o mtime na assinatura renova o cache quando o banco muda."""
    conn = load.conectar()
    try:
        return load.ler_resultados(conn)
    finally:
        conn.close()


def estilizar(fig, legenda=False):
    fig.update_layout(margin=dict(t=50, b=20, l=10, r=10), showlegend=legenda)
    return fig


mtime = load.DB_PATH.stat().st_mtime if load.DB_PATH.exists() else 0
dados = carregar(mtime)
if dados.empty:
    st.error("Banco vazio. Rode `python -m src.main` para coletar os dados primeiro.")
    st.stop()

st.title("🏎️ F1 Analytics")
st.caption(
    "Resultados históricos da Fórmula 1 (Jolpica-F1). "
    "Pergunta central: o quanto a posição de largada decide a corrida?"
)

# ---------- filtros ----------
with st.sidebar:
    st.header("Filtros")
    ano_min, ano_max = int(dados["temporada"].min()), int(dados["temporada"].max())
    if ano_min == ano_max:
        anos = (ano_min, ano_max)
        st.write(f"Temporada: {ano_min}")
    else:
        anos = st.slider("Temporadas", ano_min, ano_max, (ano_min, ano_max))
    circuito = st.selectbox(
        "Circuito", ["Todos"] + sorted(dados["circuito"].dropna().unique())
    )
    equipes_sel = st.multiselect("Equipes", sorted(dados["equipe"].dropna().unique()))

f = dados[dados["temporada"].between(*anos)]
if circuito != "Todos":
    f = f[f["circuito"] == circuito]
if equipes_sel:
    f = f[f["equipe"].isin(equipes_sel)]
if f.empty:
    st.warning("Nenhum resultado para esses filtros.")
    st.stop()

d = analysis.preparar(f)
if d.empty:
    st.warning("Sem dados de largada e chegada para esses filtros.")
    st.stop()

aba1, aba2, aba3, aba4 = st.tabs(
    ["Visão geral", "Largada x chegada", "Pilotos", "Modelo de previsão"]
)

# ---------- visão geral ----------
with aba1:
    pole = d[d["grid"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corridas", f.groupby(["temporada", "rodada"]).ngroups)
    c2.metric("Pilotos", f["piloto_id"].nunique())
    c3.metric("Equipes", f["equipe_id"].nunique())
    c4.metric(
        "Vitória saindo da pole",
        f"{pole['venceu'].mean() * 100:.1f}%" if len(pole) else "—",
    )

    esq, dir_ = st.columns(2)
    vit_pilotos = (
        f[f["posicao_final"] == 1].groupby("piloto").size().nlargest(10).sort_values()
    )
    vit_equipes = (
        f[f["posicao_final"] == 1].groupby("equipe").size().nlargest(10).sort_values()
    )
    with esq:
        if vit_pilotos.empty:
            st.info("Nenhuma vitória nesse recorte.")
        else:
            fig = px.bar(
                vit_pilotos, orientation="h", title="Pilotos com mais vitórias",
                labels={"value": "Vitórias", "piloto": ""},
                color_discrete_sequence=[VERMELHO],
            )
            st.plotly_chart(estilizar(fig))
    with dir_:
        if vit_equipes.empty:
            st.info("Nenhuma vitória nesse recorte.")
        else:
            fig = px.bar(
                vit_equipes, orientation="h", title="Equipes com mais vitórias",
                labels={"value": "Vitórias", "equipe": ""},
                color_discrete_sequence=["#15151e"],
            )
            st.plotly_chart(estilizar(fig))

# ---------- largada x chegada ----------
with aba2:
    esq, dir_ = st.columns(2)
    with esq:
        taxa = analysis.taxa_vitoria_por_grid(d).rename("Vitórias (%)").reset_index()
        fig = px.bar(
            taxa, x="grid", y="Vitórias (%)",
            title="Chance de vitória por posição de largada",
            labels={"grid": "Posição de largada"},
            color_discrete_sequence=[VERMELHO],
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(estilizar(fig))
    with dir_:
        corr = analysis.correlacao_por_temporada(d).rename("Correlação").reset_index()
        if len(corr) > 1:
            fig = px.line(
                corr, x="temporada", y="Correlação", markers=True,
                title="Quanto a largada explica a chegada (Spearman)",
                labels={"temporada": "Temporada"},
                color_discrete_sequence=["#15151e"],
            )
            st.plotly_chart(estilizar(fig))
        else:
            st.metric(
                "Correlação largada x chegada (Spearman)",
                f"{corr['Correlação'].iloc[0]:.2f}",
            )

    matriz = analysis.matriz_grid_chegada(d)
    fig = px.imshow(
        matriz, aspect="auto", color_continuous_scale="Reds",
        title="Onde cada posição de largada termina (% das largadas)",
        labels={"x": "Posição final", "y": "Posição de largada", "color": "%"},
    )
    st.plotly_chart(estilizar(fig))

    if circuito == "Todos":
        por_circuito = (
            pole.groupby("circuito")["venceu"]
            .agg(taxa="mean", corridas="count")
            .query("corridas >= 5")
        )
        if not por_circuito.empty:
            por_circuito["taxa"] = por_circuito["taxa"] * 100
            top = por_circuito.nlargest(15, "taxa").sort_values("taxa")
            fig = px.bar(
                top, x="taxa", y=top.index, orientation="h",
                hover_data=["corridas"],
                title="Onde a pole mais vira vitória (circuitos com 5+ corridas)",
                labels={"taxa": "Vitórias saindo da pole (%)", "y": ""},
                color_discrete_sequence=[VERMELHO],
            )
            st.plotly_chart(estilizar(fig))

# ---------- pilotos ----------
with aba3:
    pilotos = f["piloto"].value_counts().index.tolist()
    piloto = st.selectbox("Piloto", pilotos)
    p = f[f["piloto"] == piloto]
    p_d = d[d["piloto"] == piloto]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corridas", len(p))
    c2.metric("Vitórias", int((p["posicao_final"] == 1).sum()))
    c3.metric("Pódios", int((p["posicao_final"] <= 3).sum()))
    c4.metric("Pontos", f"{p['pontos'].sum():.0f}")

    esq, dir_ = st.columns(2)
    por_temp = (
        p.groupby("temporada")
        .agg(
            corridas=("rodada", "nunique"),
            vitorias=("posicao_final", lambda s: int((s == 1).sum())),
            podios=("posicao_final", lambda s: int((s <= 3).sum())),
            pontos=("pontos", "sum"),
            chegada_media=("posicao_final", "mean"),
        )
        .round(1)
        .reset_index()
    )
    with esq:
        fig = px.bar(
            por_temp, x="temporada", y="pontos", title="Pontos por temporada",
            labels={"temporada": "Temporada", "pontos": "Pontos"},
            color_discrete_sequence=[VERMELHO],
        )
        st.plotly_chart(estilizar(fig))
    with dir_:
        if p_d.empty:
            st.info("Sem dados de largada para esse piloto.")
        else:
            medias = (
                p_d.groupby("temporada")[["grid", "posicao_final"]]
                .mean()
                .rename(columns={"grid": "Largada média", "posicao_final": "Chegada média"})
                .reset_index()
                .melt(id_vars="temporada", var_name="Medida", value_name="Posição")
            )
            fig = px.line(
                medias, x="temporada", y="Posição", color="Medida", markers=True,
                title="Largada x chegada média por temporada",
                labels={"temporada": "Temporada"},
                color_discrete_map={
                    "Largada média": "#9a9a9a", "Chegada média": VERMELHO,
                },
            )
            fig.update_yaxes(autorange="reversed")  # posição 1 no topo
            st.plotly_chart(estilizar(fig, legenda=True))
    st.dataframe(por_temp, hide_index=True)

# ---------- modelo ----------
with aba4:
    st.caption("Esta aba não depende dos filtros da barra lateral.")
    if METRICAS.exists():
        m = pd.read_csv(METRICAS)
        media = (
            m.groupby("modelo")[["roc_auc", "avg_precision", "acertos_por_corrida"]]
            .mean()
            .round(3)
        )
        st.subheader("Média nas temporadas de teste")
        st.dataframe(media)
        fig = px.bar(
            m, x="temporada", y="acertos_por_corrida", color="modelo",
            barmode="group", color_discrete_map=CORES_MODELO,
            title="Pilotos do pódio acertados por corrida (máx. 3)",
            labels={"temporada": "Temporada", "acertos_por_corrida": "Acertos por corrida"},
        )
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[0, 3])
        st.plotly_chart(estilizar(fig, legenda=True))
        if FIG_IMPORTANCIA.exists():
            st.image(str(FIG_IMPORTANCIA), caption="Importância das variáveis")
        st.caption(
            "Avaliação walk-forward: cada temporada de teste usa um modelo treinado "
            "apenas com temporadas anteriores. Baseline: a regra 'o pódio é quem "
            "larga na frente'."
        )
    else:
        st.info("Rode `python -m src.train` para gerar as métricas do modelo.")

st.divider()
st.caption(
    "Limitações: a posição final inclui abandonos (classificados pelas voltas "
    "completadas) e largadas dos boxes ficam fora das análises de largada."
)
