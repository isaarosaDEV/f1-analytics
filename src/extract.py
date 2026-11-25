"""Coleta de resultados de corridas na API Jolpica-F1 (compatível com Ergast)."""
import logging
import time

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
TAMANHO_PAGINA = 100
PAUSA_ENTRE_REQUISICOES = 1.0  # segundos; respeita os limites da API
HEADERS = {"User-Agent": "f1-analytics-portfolio/1.0"}


def _get(url, params, tentativas=4, timeout=30):
    """GET com novas tentativas e espera maior quando a API pede calma (429)."""
    for n in range(1, tentativas + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429:
                espera = 5 * n
                log.warning("Limite da API atingido; aguardando %ds.", espera)
                time.sleep(espera)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning("Tentativa %d/%d falhou (%s): %s", n, tentativas, url, e)
            time.sleep(2 * n)
    raise RuntimeError(f"Não foi possível obter {url}")


def buscar_corridas_temporada(ano):
    """Devolve a lista de corridas (com resultados) de uma temporada.

    A API pagina por linhas de resultado, então uma mesma corrida pode vir
    dividida em duas páginas. O transform achata tudo em linhas de resultado,
    e o load grava de forma idempotente, então isso não causa duplicatas.
    """
    corridas, offset = [], 0
    while True:
        dados = _get(
            f"{BASE_URL}/{ano}/results.json",
            {"limit": TAMANHO_PAGINA, "offset": offset},
        )["MRData"]
        corridas.extend(dados["RaceTable"]["Races"])
        total = int(dados.get("total", 0))
        passo = int(dados.get("limit", TAMANHO_PAGINA)) or TAMANHO_PAGINA
        offset += passo
        if offset >= total:
            break
        time.sleep(PAUSA_ENTRE_REQUISICOES)
    return corridas
