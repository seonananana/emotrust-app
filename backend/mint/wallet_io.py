# mint/wallet_io.py
from pathlib import Path
import json

WALLET_JSON = Path(__file__).with_name("wallet.json")

def load_wallet():
    """
    mint/wallet.json에서 (private_key, address) 로드.
    없으면 (None, None) 반환.
    """
    if not WALLET_JSON.exists():
        return None, None
    obj = json.loads(WALLET_JSON.read_text())
    pk = obj.get("private_key") or obj.get("PRIVATE_KEY")
    addr = obj.get("address") or obj.get("PUBLIC_ADDRESS")
    if pk and not pk.startswith("0x"):
        pk = "0x" + pk
    return pk, addr
