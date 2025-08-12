# backend/mint/mint.py
import os, json, re
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]  # backend/
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# --- Feature flag: disable mint ---
DISABLE_MINT = os.getenv("EMOTRUST_DISABLE_MINT") == "1"
if DISABLE_MINT:
    # 민팅 비활성화 모드: 서버만 뜨게 안전한 스텁 제공
    def send_mint(*args, **kwargs):
        raise RuntimeError("Minting disabled (set EMOTRUST_DISABLE_MINT=0 to enable).")
    def wait_token_id(*args, **kwargs):
        raise RuntimeError("Minting disabled (set EMOTRUST_DISABLE_MINT=0 to enable).")
else:
    # --- Network / Provider ---
    RPC_URL = os.getenv("RPC_URL", "").strip()
    if not RPC_URL:
        raise RuntimeError("RPC_URL이 없습니다. backend/.env에 Alchemy/Infura HTTPS URL을 넣으세요.")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    # --- Chain ---
    CHAIN_ID = int(os.getenv("CHAIN_ID", "11155111"))  # default: Sepolia

    # --- Wallet ---
    PRIVATE_KEY = (os.getenv("PRIVATE_KEY") or "").strip()
    PUBLIC_ADDRESS = (os.getenv("PUBLIC_ADDRESS") or "").strip()

    def _is_hex_addr(s: str) -> bool:
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", s or ""))

    if not PRIVATE_KEY or not PUBLIC_ADDRESS:
        raise RuntimeError("PRIVATE_KEY/PUBLIC_ADDRESS가 없습니다. `python mint/gen_wallet.py`로 생성하세요.")

    if "," in PUBLIC_ADDRESS or not _is_hex_addr(PUBLIC_ADDRESS) or not Web3.is_address(PUBLIC_ADDRESS):
        raise ValueError(f"Invalid PUBLIC_ADDRESS: {PUBLIC_ADDRESS!r}")

    PUBLIC_ADDR = Web3.to_checksum_address(PUBLIC_ADDRESS)

    # --- Contract address (env -> txt fallback, but strictly validated) ---
    contract_addr = (os.getenv("CONTRACT_ADDRESS") or "").strip()
    if not contract_addr:
        p = Path(__file__).resolve().parent / "contract_address.txt"
        if p.exists():
            raw = p.read_text(encoding="utf-8").strip()
