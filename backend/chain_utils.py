from __future__ import annotations
import os, json, re
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

RPC_URL = (os.getenv("RPC_URL") or "").strip()
if not RPC_URL:
    raise RuntimeError("RPC_URL이 없습니다. .env에 설정하세요.")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

addr = (os.getenv("CONTRACT_ADDRESS") or "").strip()
if not re.fullmatch(r"0x[a-fA-F0-9]{40}", addr or ""):
    raise RuntimeError("CONTRACT_ADDRESS가 유효하지 않습니다(0x40hex)")

ABI_PATH = ROOT / "mint" / "abi" / "EmpathyNFT.abi.json"
CONTRACT = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=json.loads(ABI_PATH.read_text(encoding="utf-8")))

def nft_balance_of(address: str) -> int:
    return int(CONTRACT.functions.balanceOf(Web3.to_checksum_address(address)).call())
