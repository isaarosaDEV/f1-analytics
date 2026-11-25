"""Banco SQLite: gravação idempotente e leitura dos resultados."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "f1.db"

COLUNAS = [
    "temporada", "rodada", "corrida", "data", "circuito_id", "circuito", "pais",
    "piloto_id", "piloto", "equipe_id", "equipe", "grid", "posicao_final",
    "posicao_texto", "pontos", "voltas", "status",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS resultados (
    temporada INTEGER NOT NULL,
    rodada INTEGER NOT NULL,
    corrida TEXT,
    data TEXT,
    circuito_id TEXT,
    circuito TEXT,
    pais TEXT,
    piloto_id TEXT NOT NULL,
    piloto TEXT,
    equipe_id TEXT,
    equipe TEXT,
    grid INTEGER,
    posicao_final INTEGER,
    posicao_texto TEXT,
    pontos REAL,
    voltas INTEGER,
    status TEXT,
    PRIMARY KEY (temporada, rodada, piloto_id)
);
"""


def conectar(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def carregar(conn, df):
    """Grava os resultados; rodar duas vezes não duplica nada."""
    if df.empty:
        return 0
    sub = df[COLUNAS]
    limpo = sub.astype(object).where(sub.notna(), None)
    sql = (
        f"INSERT OR REPLACE INTO resultados ({', '.join(COLUNAS)}) "
        f"VALUES ({', '.join(':' + c for c in COLUNAS)})"
    )
    conn.executemany(sql, limpo.to_dict("records"))
    conn.commit()
    return len(limpo)


def ler_resultados(conn):
    return pd.read_sql_query("SELECT * FROM resultados", conn)
