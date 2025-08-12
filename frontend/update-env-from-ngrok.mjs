// Node 18+ 가정 (global fetch 사용)
import { writeFileSync } from 'fs';

const NGROK_API = 'http://127.0.0.1:4040/api/tunnels';

try {
  const res = await fetch(NGROK_API);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const https = data.tunnels.find(t => t.proto === 'https')?.public_url;

  if (!https) throw new Error('https 터널을 찾지 못했습니다. (ngrok이 켜져있는지 확인)');

  writeFileSync('./.env', `EXPO_PUBLIC_API_BASE_URL=${https}\n`);
  console.log('[env] EXPO_PUBLIC_API_BASE_URL =', https);
} catch (e) {
  console.error('[env] ngrok 주소 갱신 실패:', e.message);
  process.exit(1);
}
