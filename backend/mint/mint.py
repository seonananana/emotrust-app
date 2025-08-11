import os, json, base64
from web3 import Web3

RPC_URL      = os.getenv("RPC_URL")
CHAIN_ID     = int(os.getenv("CHAIN_ID", "11155111"))  # Sepolia
PRIVATE_KEY  = os.getenv("PRIVATE_KEY")
PUBLIC_ADDR  = Web3.to_checksum_address(os.getenv("PUBLIC_ADDRESS"))
CONTRACT     = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))
ABI_PATH     = os.getenv("ERC721_ABI_PATH", "./mint/abi/EmpathyNFT.abi.json")
MINT_FN_NAME = os.getenv("MINT_FN_NAME", "safeMint")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
with open(ABI_PATH, "r", encoding="utf-8") as f:
    abi = json.load(f)
contract = w3.eth.contract(address=CONTRACT, abi=abi)

def _data_uri(meta: dict) -> str:
    j = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    b64 = base64.b64encode(j.encode()).decode()
    return f"data:application/json;base64,{b64}"

def send_mint(to_addr: str, meta: dict) -> str:
    to = Web3.to_checksum_address(to_addr)
    uri = _data_uri(meta)
    fn = getattr(contract.functions, MINT_FN_NAME)(to, uri)

    nonce = w3.eth.get_transaction_count(PUBLIC_ADDR)
    tx = fn.build_transaction({
        "from": PUBLIC_ADDR,
        "nonce": nonce,
        "chainId": CHAIN_ID,
        "type": 2,  # EIP-1559
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
    })
    tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)

    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    return tx_hash.hex()

def wait_token_id(tx_hash_hex: str, timeout: int = 180):
    rcpt = w3.eth.wait_for_transaction_receipt(tx_hash_hex, timeout=timeout)
    # Transfer 이벤트에서 tokenId 추출
    try:
        events = contract.events.Transfer().process_receipt(rcpt)
        if events:
            return int(events[0]["args"]["tokenId"]), rcpt
    except Exception:
        pass
    return None, rcpt
