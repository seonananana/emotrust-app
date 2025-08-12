require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config({ path: "./.env" });

const { RPC_URL, PRIVATE_KEY } = process.env;
const hasPK = /^0x[0-9a-fA-F]{64}$/.test((PRIVATE_KEY || "").trim());

const networks = {
  hardhat: {},                               // 로컬 in-memory 체인
  localhost: { url: "http://127.0.0.1:8545" } // 로컬 노드 접속용
};

// PRIVATE_KEY가 올바를 때만 sepolia 네트워크를 활성화
if (hasPK && RPC_URL) {
  networks.sepolia = {
    url: RPC_URL,
    accounts: [PRIVATE_KEY],
    chainId: 11155111
  };
}

module.exports = {
  solidity: "0.8.24",
  defaultNetwork: "hardhat",  // ← 반드시 hardhat
  networks
};
