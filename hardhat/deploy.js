const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

async function main() {
  const Empathy = await hre.ethers.getContractFactory("EmpathyNFT");
  const c = await Empathy.deploy();
  await c.deployed();

  console.log("EmpathyNFT deployed to:", c.address);

  // ABI 추출 → backend로 복사
  const artifact = await hre.artifacts.readArtifact("EmpathyNFT");
  const abiPath = path.join(__dirname, "../backend/mint/abi/EmpathyNFT.abi.json");
  fs.mkdirSync(path.dirname(abiPath), { recursive: true });
  fs.writeFileSync(abiPath, JSON.stringify(artifact.abi, null, 2));

  // 주소 저장(백엔드 .env 채울 때 사용)
  fs.writeFileSync(path.join(__dirname, "./EmpathyNFT.address"), c.address);
}

main().catch((e) => { console.error(e); process.exit(1); });
