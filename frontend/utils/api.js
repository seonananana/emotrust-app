// utils/api.js
const BASE_URL = 'http://localhost:8000'; // ngrok 연결 시 수정

export async function fetchJSON(path, body = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(body).toString(),
  });
  return res.json();
}
