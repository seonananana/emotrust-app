# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sqlite3, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "emotrust_sim.db"

# 네임스페이스 분리(원하면 여러 환경 데이터 분리 가능)
NAMESPACE = os.getenv("SIM_CHAIN_NAMESPACE", "default")

def _conn():
    cx = sqlite3.connect(DB_PATH)
    cx.execute("""
    CREATE TABLE IF NOT EXISTS sim_mints (
        namespace TEXT NOT NULL,
        addr TEXT NOT NULL,
        token_id INTEGER NOT NULL,
        ts INTEGER NOT NULL
    )
    """)
    return cx

def sim_mint(to_addr: str) -> tuple[str, int]:
    with _conn() as cx:
        cur = cx.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM sim_mints WHERE namespace=? AND addr=?",
            (NAMESPACE, to_addr),
        )
        n = cur.fetchone()[0]
        token_id = n + 1
        cur.execute(
            "INSERT INTO sim_mints(namespace, addr, token_id, ts) VALUES(?,?,?,?)",
            (NAMESPACE, to_addr, token_id, int(time.time())),
        )
    # 가짜 tx_hash 생성(주소+토큰ID+타임스탬프)
    h = hashlib.sha256(f"{to_addr}:{token_id}:{time.time()}".encode()).hexdigest()
    tx_hash = "0x" + h
    return tx_hash, token_id

def sim_balance_of(addr: str) -> int:
    with _conn() as cx:
        cur = cx.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM sim_mints WHERE namespace=? AND addr=?",
            (NAMESPACE, addr),
        )
        return int(cur.fetchone()[0])
