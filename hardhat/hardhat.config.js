require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config({ path: "./.env" });

const RPC_URL = process.env.RPC_URL || "";
const PRIVATE_KEY = (process.env.PRIVATE_KEY || "").trim();
const hasPK = /^0x[0-9a-fA-F]{64}$/.test(PRIVATE_KEY);

const networks = {
  hardhat: {},
  localhost: { url: "http://127.0.0.1:8545" },
};
if (hasPK && RPC_URL) {
  networks.sepolia = { url: RPC_URL, accounts: [PRIVATE_KEY], chainId: 11155111 };
}

module.exports = {
  solidity: { version: "0.8.24", settings: { optimizer: { enabled: true, runs: 200 } } },
  defaultNetwork: "hardhat",
  paths: { sources: "contracts", tests: "test", cache: "cache", artifacts: "artifacts" },
  networks,
};
