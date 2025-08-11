# backend/mint/gen_wallet.py
from pathlib import Path
from dotenv import load_dotenv, set_key
from eth_account import Account
import argparse, os

def main(force: bool):
    # backend/.env 경로 지정 (mint 폴더의 부모가 backend)
    env_file = Path(__file__).resolve().parents[1] / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)

    load_dotenv(dotenv_path=env_file)
    pub, priv = os.getenv("PUBLIC_ADDRESS"), os.getenv("PRIVATE_KEY")

    if pub and priv and not force:
        print("이미 키가 있어서 새로 만들지 않음.")
        print("PUBLIC_ADDRESS =", pub)
        print("env:", env_file)
        return

    acct = Account.create()
    set_key(str(env_file), "PUBLIC_ADDRESS", acct.address)
    set_key(str(env_file), "PRIVATE_KEY",  acct.key.hex())
    print(("덮어쓰기 완료" if pub and priv else "생성 & 저장 완료"), "->", env_file)
    print("PUBLIC_ADDRESS =", acct.address)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="기존 키가 있어도 새로 생성해 덮어쓰기")
    args = ap.parse_args()
    main(force=args.force)
