const fs = require("fs");
const path = require("path");
const { ethers } = require("hardhat");

async function main() {
  const EmpathyNFT = await ethers.getContractFactory("EmpathyNFT");
  const nft = await EmpathyNFT.deploy();
  await nft.waitForDeployment();

  const addr = await nft.getAddress();
  console.log("Deployed to:", addr);

  // ABI 파일을 백엔드로 복사 (있으면 덮어씀)
  const artifact = await hre.artifacts.readArtifact("EmpathyNFT");
  const outDir = path.join(__dirname, "..", "backend", "mint", "abi");
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "EmpathyNFT.abi.json"), JSON.stringify(artifact.abi, null, 2));

  // 주소 기록 (선택)
  const addrFile = path.join(__dirname, "..", "backend", "mint", "contract_address.txt");
  fs.writeFileSync(addrFile, addr);
  console.log("ABI/주소 저장:", outDir, addrFile);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
