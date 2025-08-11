const fs = require("fs");
const path = require("path");

async function main() {
  const EmpathyNFT = await ethers.deployContract("EmpathyNFT");
  await EmpathyNFT.waitForDeployment();
  const addr = await EmpathyNFT.getAddress();
  console.log("EmpathyNFT deployed to:", addr);

  // 주소 저장 → backend에서 자동 사용
  const out = path.join(__dirname, "..", "backend", "mint", "contract_address.txt");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, addr, "utf8");

  // ABI 복사
  const artifact = await artifacts.readArtifact("EmpathyNFT");
  const abiOut = path.join(__dirname, "..", "backend", "mint", "abi", "EmpathyNFT.abi.json");
  fs.mkdirSync(path.dirname(abiOut), { recursive: true });
  fs.writeFileSync(abiOut, JSON.stringify(artifact.abi, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
