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
            # 'CONTRACT_ADDRESS=...' 형식이든 아니든, 첫 0x40hex만 추출
            m = re.search(r"0x[a-fA-F0-9]{40}", raw)
            contract_addr = m.group(0) if m else ""

    if not contract_addr:
        raise RuntimeError("CONTRACT_ADDRESS가 없습니다. .env 또는 mint/contract_address.txt에 0x로 시작하는 40자리 주소만 넣으세요.")
    if "," in contract_addr:
        raise ValueError(f"CONTRACT_ADDRESS looks like event topics, not an address: {contract_addr!r}")
    if not _is_hex_addr(contract_addr) or not Web3.is_address(contract_addr):
        raise ValueError(f"Invalid CONTRACT_ADDRESS format: {contract_addr!r}")

    CONTRACT = Web3.to_checksum_address(contract_addr)

    # --- ABI ---
    default_abi = Path(__file__).resolve().parent / "abi" / "EmpathyNFT.abi.json"
    ABI_PATH = Path(os.getenv("ERC721_ABI_PATH", str(default_abi))).resolve()
    if not ABI_PATH.exists():
        raise FileNotFoundError(f"ERC721_ABI_PATH 파일을 찾을 수 없습니다: {ABI_PATH}")
    with open(ABI_PATH, "r", encoding="utf-8") as f:
        abi = json.load(f)

    MINT_FN_NAME = os.getenv("MINT_FN_NAME", "safeMint")
    contract = w3.eth.contract(address=CONTRACT, abi=abi)

    def _data_uri(meta: dict) -> str:
        j = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
        import base64 as _b64
        return f"data:application/json;base64,{_b64.b64encode(j.encode()).decode()}"

    def send_mint(to_addr: str, meta: dict) -> str:
        if "," in to_addr or not _is_hex_addr(to_addr) or not Web3.is_address(to_addr):
            raise ValueError(f"Invalid recipient address: {to_addr!r}")

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

        # 여유 버퍼 but 과도 방지
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(min(est * 1.2, est + 150_000))

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
