// hardhat/hardhat.config.js
const path = require("path");
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config({ path: path.join(__dirname, ".env") });

const RPC_URL = process.env.RPC_URL || "";
const PRIVATE_KEY = (process.env.PRIVATE_KEY || "").trim();
const hasPK = /^0x[0-9a-fA-F]{64}$/.test(PRIVATE_KEY);

// 기본 네트워크들
const networks = {
  hardhat: {},                           // 로컬 in-memory 체인
  localhost: { url: "http://127.0.0.1:8545" }, // 로컬 노드 접속
};

// PRIVATE_KEY와 RPC_URL이 제대로 있을 때만 sepolia 활성화
if (hasPK && RPC_URL) {
  networks.sepolia = {
    url: RPC_URL,
    accounts: [PRIVATE_KEY],
    chainId: 11155111,
  };
}

module.exports = {
  solidity: {
    version: "0.8.24",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  defaultNetwork: "hardhat",

  // ★ 루트(hardhat 폴더)를 소스 경로로 지정 → EmpathyNFT.sol 이 루트에 있어도 컴파일됨
  paths: {
    sources: __dirname,     // ← 중요
    tests: "test",
    cache: "cache",
    artifacts: "artifacts",
  },

  networks,
};
