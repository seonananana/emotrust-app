# simulate_chain.py
# -*- coding: utf-8 -*-
import random
import string

# 시뮬 블록체인 환경 저장
_SIM_CHAIN = {
    "next_token_id": 1,
    "balances": {}  # address -> token_count
}

def _random_hex(n):
    return "0x" + "".join(random.choices("0123456789abcdef", k=n))

def sim_mint(to_address: str):
    """
    가짜 민팅: 토큰 ID 발급 + tx_hash 생성
    """
    global _SIM_CHAIN
    token_id = _SIM_CHAIN["next_token_id"]
    _SIM_CHAIN["next_token_id"] += 1

    # 소유권 업데이트
    _SIM_CHAIN["balances"][to_address.lower()] = _SIM_CHAIN["balances"].get(to_address.lower(), 0) + 1

    # 랜덤 트랜잭션 해시 생성
    tx_hash = _random_hex(64)

    return tx_hash, token_id

def sim_balance_of(address: str) -> int:
    """
    시뮬 모드에서의 토큰 개수 반환
    """
    return _SIM_CHAIN["balances"].get(address.lower(), 0)
