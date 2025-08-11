# backend/mint/mint.py
import os, json, base64
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # backend/
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

RPC_URL = os.getenv("RPC_URL")
if not RPC_URL:
    raise RuntimeError("RPC_URL이 없습니다. backend/.env에 Alchemy/Infura HTTPS URL을 넣으세요.")
w3 = Web3(Web3.HTTPProvider(RPC_URL))

CHAIN_ID = int(os.getenv("CHAIN_ID", "11155111"))

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
PUBLIC_ADDRESS = os.getenv("PUBLIC_ADDRESS")
if not PRIVATE_KEY or not PUBLIC_ADDRESS:
    raise RuntimeError("PRIVATE_KEY/PUBLIC_ADDRESS가 없습니다. `python mint/gen_wallet.py`로 생성하세요.")
PUBLIC_ADDR = Web3.to_checksum_address(PUBLIC_ADDRESS)

# ✅ .env 비어 있으면 contract_address.txt 폴백
contract_addr = (os.getenv("CONTRACT_ADDRESS") or "").strip()
if not contract_addr:
    p = Path(__file__).resolve().parent / "contract_address.txt"
    if p.exists():
        contract_addr = p.read_text().strip()
if not contract_addr:
    raise RuntimeError("CONTRACT_ADDRESS가 없습니다. .env에 넣거나 mint/contract_address.txt 추가하세요.")
CONTRACT = Web3.to_checksum_address(contract_addr)

# ABI
default_abi = Path(__file__).resolve().parent / "abi" / "EmpathyNFT.abi.json"
ABI_PATH = os.getenv("ERC721_ABI_PATH", str(default_abi))
with open(ABI_PATH, "r", encoding="utf-8") as f:
    abi = json.load(f)

MINT_FN_NAME = os.getenv("MINT_FN_NAME", "safeMint")
contract = w3.eth.contract(address=CONTRACT, abi=abi)

def _data_uri(meta: dict) -> str:
    j = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    import base64 as _b64
    return f"data:application/json;base64,{_b64.b64encode(j.encode()).decode()}"

def send_mint(to_addr: str, meta: dict) -> str:
    to = Web3.to_checksum_address(to_addr)
    uri = _data_uri(meta)
    fn = getattr(contract.functions, MINT_FN_NAME)(to, uri)

    nonce = w3.eth.get_transaction_count(PUBLIC_ADDR)
    max_fee_gwei = float(os.getenv("MAX_FEE_GWEI", "2"))
    max_prio_gwei = float(os.getenv("MAX_PRIORITY_GWEI", "1"))

    tx = fn.build_transaction({
        "from": PUBLIC_ADDR,
        "nonce": nonce,
        "chainId": CHAIN_ID,
        "type": 2,
        "maxFeePerGas": w3.to_wei(max_fee_gwei, "gwei"),
        "maxPriorityFeePerGas": w3.to_wei(max_prio_gwei, "gwei"),
    })
    tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    return tx_hash.hex()

def wait_token_id(tx_hash_hex: str, timeout: int = 180):
    rcpt = w3.eth.wait_for_transaction_receipt(tx_hash_hex, timeout=timeout)
    try:
        ev = contract.events.Transfer().process_receipt(rcpt)
        if ev:
            return int(ev[0]["args"]["tokenId"]), rcpt
    except Exception:
        pass
    return None, rcpt
