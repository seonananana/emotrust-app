import { writeFileSync } from 'fs';

const res = await fetch('http://127.0.0.1:4040/api/tunnels');
const data = await res.json();
const https = data.tunnels.find(t => t.proto === 'https')?.public_url;

if (!https) {
  console.error('ngrok https 터널을 못 찾았습니다. ngrok이 실행 중인지 확인하세요.');
  process.exit(1);
}
writeFileSync('./.env', `EXPO_PUBLIC_API_BASE_URL=${https}\n`);
console.log('frontend/.env 갱신:', https);
