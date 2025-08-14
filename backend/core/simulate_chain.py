# backend/core/simulate_chain.py

import random

_SIM_CHAIN = {
    "next_token_id": 1,
    "balances": {}
}

def _random_hex(n):
    return "0x" + "".join(random.choices("0123456789abcdef", k=n))

def sim_mint(to_address: str):
    token_id = _SIM_CHAIN["next_token_id"]
    _SIM_CHAIN["next_token_id"] += 1

    _SIM_CHAIN["balances"][to_address.lower()] = _SIM_CHAIN["balances"].get(to_address.lower(), 0) + 1

    tx_hash = _random_hex(64)
    return tx_hash, token_id

def sim_balance_of(address: str) -> int:
    return _SIM_CHAIN["balances"].get(address.lower(), 0)
