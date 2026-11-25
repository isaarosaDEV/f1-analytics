"""Transforma o JSON da API em uma tabela plana (uma linha por piloto por corrida)."""
import pandas as pd


def _int(valor, padrao=None):
    try:
        return int(valor)
    except (TypeError, ValueError):
        return padrao


def _float(valor, padrao=None):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def achatar_resultados(corridas):
    linhas = []
    for c in corridas:
        circuito = c.get("Circuit", {})
        local = circuito.get("Location", {})
        for r in c.get("Results", []):
            piloto = r.get("Driver", {})
            equipe = r.get("Constructor", {})
            linhas.append(
                {
                    "temporada": _int(c.get("season")),
                    "rodada": _int(c.get("round")),
                    "corrida": c.get("raceName"),
                    "data": c.get("date"),
                    "circuito_id": circuito.get("circuitId"),
                    "circuito": circuito.get("circuitName"),
                    "pais": local.get("country"),
                    "piloto_id": piloto.get("driverId"),
                    "piloto": f"{piloto.get('givenName', '')} {piloto.get('familyName', '')}".strip(),
                    "equipe_id": equipe.get("constructorId"),
                    "equipe": equipe.get("name"),
                    "grid": _int(r.get("grid")),
                    "posicao_final": _int(r.get("position")),
                    "posicao_texto": r.get("positionText"),
                    "pontos": _float(r.get("points"), 0.0),
                    "voltas": _int(r.get("laps")),
                    "status": r.get("status"),
                }
            )
    return pd.DataFrame(linhas)
