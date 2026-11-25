"""Coleta os resultados, grava no SQLite e gera a análise de largada x chegada."""
import argparse
import logging
import time
from datetime import date
from pathlib import Path

from src import analysis, extract, load, transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("f1")

FIGURAS = Path(__file__).resolve().parent.parent / "reports" / "figures"


def main():
    p = argparse.ArgumentParser(description="Pipeline de dados de F1")
    p.add_argument("--inicio", type=int, default=2010, help="primeira temporada")
    p.add_argument("--fim", type=int, default=date.today().year, help="última temporada")
    p.add_argument("--so-analise", action="store_true", help="pula a coleta")
    args = p.parse_args()

    conn = load.conectar()

    if not args.so_analise:
        for ano in range(args.inicio, args.fim + 1):
            corridas = extract.buscar_corridas_temporada(ano)
            df = transform.achatar_resultados(corridas)
            n = load.carregar(conn, df)
            log.info("Temporada %d: %d resultados gravados.", ano, n)
            time.sleep(extract.PAUSA_ENTRE_REQUISICOES)

    dados = load.ler_resultados(conn)
    conn.close()
    if dados.empty:
        log.error("Banco vazio. Rode a coleta primeiro.")
        raise SystemExit(1)

    d = analysis.preparar(dados)
    for chave, valor in analysis.resumo(d).items():
        log.info("%s: %s", chave, valor)
    analysis.gerar_figuras(d, FIGURAS)
    log.info("Figuras salvas em %s", FIGURAS)


if __name__ == "__main__":
    main()
