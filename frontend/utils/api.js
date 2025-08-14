// utils/api.js
import axios from 'axios';

const BASE = process.env.EXPO_PUBLIC_API_BASE_URL || 'https://your-ngrok-url.ngrok-free.app';

export async function fetchAnalyzeAndMint({ title, content, gate }) {
  const form = new FormData();
  form.append('title', title);
  form.append('content', content);
  form.append('gate', gate);
  form.append('w_acc', 0.5);
  form.append('w_sinc', 0.5);
  form.append('denom_mode', 'all');

  const { data } = await axios.post(`${BASE}/analyze-and-mint`, form);
  return data;
}

export async function fetchPosts() {
  const { data } = await axios.get(`${BASE}/posts`);
  return data.items || [];
}
