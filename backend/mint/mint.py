# -*- coding: utf-8 -*-
"""Safe mint pipeline with env-based hard stop and strict validation."""
from __future__ import annotations
import json, os, re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

ROOT = Path(__file__).resolve().parents[1]  # backend/
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ---------- env & guards ----------
DISABLE = os.getenv("EMOTRUST_DISABLE_MINT", "0").strip() == "1"
RPC_URL = (os.getenv("RPC_URL") or "").strip()
PRIVATE_KEY = (os.getenv("PRIVATE_KEY") or "").strip()
PUBLIC_ADDRESS = (os.getenv("PUBLIC_ADDRESS") or "").strip()
CHAIN_ID = int(os.getenv("CHAIN_ID", "0") or 0)

if not RPC_URL:
    raise RuntimeError("RPC_URL이 없습니다. backend/.env에 설정하세요.")

if DISABLE:
    # 체인 접속 자체를 막아 서버 기동만 허용
    class _Disabled:
        def __getattr__(self, _):
            raise RuntimeError("Minting disabled(EMOTRUST_DISABLE_MINT=1)")
    w3 = _Disabled()  # type: ignore
else:
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    # 일부 테스트넷은 POA 미들웨어 필요
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    if not w3.is_connected():
        raise RuntimeError("Web3 RPC에 연결되지 않았습니다: RPC_URL 확인")

# ---------- contract load ----------
_ABI_PATH = Path(__file__).resolve().parent / "abi" / "EmpathyNFT.abi.json"
if not _ABI_PATH.exists():
    raise RuntimeError(f"ABI 파일이 없습니다: {_ABI_PATH}")
_ABI = json.loads(_ABI_PATH.read_text(encoding="utf-8"))

def _clean_addr(s: str) -> str:
    m = re.search(r"0x[a-fA-F0-9]{40}", s or "")
    return m.group(0) if m else ""

def _load_contract_address() -> str:
    env_addr = (os.getenv("CONTRACT_ADDRESS") or "").strip()
    if env_addr:
        return env_addr
    p = Path(__file__).resolve().parent / "contract_address.txt"
    if p.exists():
        return _clean_addr(p.read_text(encoding="utf-8", errors="ignore"))
    return ""

CONTRACT_ADDRESS = _load_contract_address()
if not re.fullmatch(r"0x[a-fA-F0-9]{40}", CONTRACT_ADDRESS or ""):
    raise RuntimeError("컨트랙트 주소가 유효하지 않습니다. .env의 CONTRACT_ADDRESS 또는 mint/contract_address.txt에 0x40hex로 설정")

if DISABLE:
    CONTRACT = None  # type: ignore
else:
    CONTRACT = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=_ABI,
    )

# ---------- validation ----------
if not DISABLE:
    if not re.fullmatch(r"0x[a-fA-F0-9]{64}", PRIVATE_KEY or ""):
        raise ValueError("PRIVATE_KEY 형식 오류: 0x + 64hex")
    if not Web3.is_address(PUBLIC_ADDRESS or ""):
        raise ValueError(f"PUBLIC_ADDRESS 형식 오류: {PUBLIC_ADDRESS}")
    if not CHAIN_ID:
        raise RuntimeError("CHAIN_ID가 설정되지 않았습니다.")

@dataclass
class MintResult:
    tx_hash: str
    token_id: Optional[int] = None

# ---------- core ops ----------

def send_mint(to_address: str) -> MintResult:
    if DISABLE:
        raise RuntimeError("Minting disabled(EMOTRUST_DISABLE_MINT=1)")
    if not Web3.is_address(to_address):
        raise ValueError(f"수신 주소가 유효하지 않습니다: {to_address}")

    nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(PUBLIC_ADDRESS))
    tx = CONTRACT.functions.safeMint(Web3.to_checksum_address(to_address)).build_transaction({
        "chainId": CHAIN_ID,
        "from": Web3.to_checksum_address(PUBLIC_ADDRESS),
        "nonce": nonce,
        # 가스 전략은 네트워크에 맞게 조정 가능
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
    })

    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    return MintResult(tx_hash=tx_hash.hex())


def wait_token_id(tx_hash_hex: str, timeout: int = 120) -> MintResult:
    if DISABLE:
        raise RuntimeError("Minting disabled(EMOTRUST_DISABLE_MINT=1)")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash_hex, timeout=timeout)
    # ERC721 Transfer(0xddf252ad...) 로그에서 tokenId 추출
    token_id = None
    for log in receipt.logs:
        try:
            evt = CONTRACT.events.Transfer().process_log(log)
            token_id = int(evt["args"]["tokenId"])  # type: ignore
            break
        except Exception:
            continue
    return MintResult(tx_hash=tx_hash_hex, token_id=token_id)
