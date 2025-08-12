// hardhat/deploy.js
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  // 컨트랙트 이름은 .sol 안의 실제 컨트랙트명과 같아야 함
  const Factory = await hre.ethers.getContractFactory("EmpathyNFT");

  // 생성자 인자 있으면 맞춰서 넣고, 없으면 deploy()만 호출
  const name = process.env.NFT_NAME || "EmpathyNFT";
  const symbol = process.env.NFT_SYMBOL || "EMO";
  const contract = await Factory.deploy(name, symbol);

  await contract.waitForDeployment();
  const address = await contract.getAddress();
  console.log("EmpathyNFT deployed to:", address);

  // ABI 추출해서 backend로 복사
  const artifact = await hre.artifacts.readArtifact("EmpathyNFT");
  const abiDir = path.join(__dirname, "..", "backend", "mint", "abi");
  fs.mkdirSync(abiDir, { recursive: true });
  fs.writeFileSync(
    path.join(abiDir, "EmpathyNFT.abi.json"),
    JSON.stringify(artifact.abi, null, 2)
  );

  // 주소도 backend에 기록 (백엔드가 폴백으로 읽음)
  const addrFile = path.join(__dirname, "..", "backend", "mint", "contract_address.txt");
  fs.mkdirSync(path.dirname(addrFile), { recursive: true });
  fs.writeFileSync(addrFile, address);

  console.log("✅ ABI / 주소를 backend/mint/ 에 저장 완료");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
