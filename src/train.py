"""Treina e avalia os modelos de previsão de pódio, e gera tabela e gráficos."""
import argparse
import logging
from pathlib import Path

from src import features, load, model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("f1.train")

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "reports" / "figures"


def main():
    p = argparse.ArgumentParser(description="Modelo de previsão de pódio na F1")
    p.add_argument("--n-anos-teste", type=int, default=3,
                   help="quantas temporadas finais usar como teste")
    p.add_argument("--min-corridas", type=int, default=15,
                   help="mínimo de corridas para a temporada contar como completa")
    args = p.parse_args()

    conn = load.conectar()
    dados = load.ler_resultados(conn)
    conn.close()
    if dados.empty:
        raise SystemExit("Banco vazio. Rode a coleta (src.main) primeiro.")

    d = features.adicionar_features(features.preparar_modelo(dados))
    anos = model.anos_completos(d, args.min_corridas)[-args.n_anos_teste:]
    if len(anos) == 0 or d["temporada"].nunique() < 3:
        raise SystemExit(
            "Poucos dados. Colete mais temporadas (por exemplo --inicio 2010) e tente de novo."
        )

    log.info("Teste nas temporadas %s; treino sempre com os anos anteriores.", anos)
    res = model.avaliar_walk_forward(d, anos)
    resumo = (
        res.groupby("modelo")[["roc_auc", "avg_precision", "acertos_por_corrida"]]
        .mean()
        .round(3)
        .reindex(model.ORDEM)
    )
    print("\nMédia nas temporadas de teste:\n")
    print(resumo.to_string())
    print()

    (RAIZ / "reports").mkdir(exist_ok=True)
    res.round(4).to_csv(RAIZ / "reports" / "metricas_por_temporada.csv", index=False)

    imp = model.importancia_variaveis(d, anos[-1])
    model.gerar_figuras(res, imp, FIGURAS)
    log.info("Métricas e figuras salvas em %s", RAIZ / "reports")


if __name__ == "__main__":
    main()
