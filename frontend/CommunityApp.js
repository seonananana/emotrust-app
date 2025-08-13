// CommunityApp.js — refactored for: 댓글=+0.02 to credit, 좋아요=공감 토큰(시뮬 민팅)
import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  Button,
  ScrollView,
  ActivityIndicator,
  Alert,
  Linking,
  TextInput,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';

const API = (process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

// ───────────────────────── helpers ─────────────────────────
async function apiGet(path) {
  const r = await fetch(`${API}${path}`);
  const t = await r.text();
  if (!r.ok) throw new Error(t || `HTTP ${r.status}`);
  try { return JSON.parse(t); } catch { return t; }
}
async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  const t = await r.text();
  if (!r.ok) throw new Error(t || `HTTP ${r.status}`);
  try { return JSON.parse(t); } catch { return t; }
}

// ───────────────────────── UI: Card (List item) ─────────────────────────
function PostCard({ p, onOpen }) {
  const sPre = Number(p.S_pre ?? 0);
  const sEff = Number(p.S_effective ?? p.S_pre ?? 0);
  return (
    <TouchableOpacity style={s.card} onPress={onOpen} activeOpacity={0.85}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
        <Text style={s.title}>#{p.id} · {p.title || '(제목 없음)'}</Text>
        {p.created_at ? <Text style={s.time}>{new Date(p.created_at).toLocaleString()}</Text> : null}
      </View>
      <Text style={s.meta}>
        S={sPre.toFixed(2)} → S_eff={sEff.toFixed(2)} · gate={String(p.gate)} · pass={String(p.gate_pass)}
      </Text>
      <Text style={s.meta}>likes: {p.likes ?? 0}</Text>
      <Text style={s.hint}>상세에서 민팅/댓글/토큰 확인</Text>
    </TouchableOpacity>
  );
}

// ─────────────────────── UI: Detail View ───────────────────────
function PostDetail({ postId, onBack }) {
  const [detail, setDetail] = useState(null);
  const [comments, setComments] = useState([]);
  const [cmt, setCmt] = useState('');
  const [addr, setAddr] = useState(''); // 좋아요(공감 토큰) 수령 주소(옵션)
  const [busyCmt, setBusyCmt] = useState(false);
  const [busyLike, setBusyLike] = useState(false);
  const [loading, setLoading] = useState(true);

  async function loadAll() {
    setLoading(true);
    try {
      const [d, c] = await Promise.all([
        apiGet(`/posts/${postId}`),
        apiGet(`/posts/${postId}/comments`)
      ]);
      setDetail(d);
      setComments(c.items || []);
    } catch (e) {
      Alert.alert('로드 오류', String(e).slice(0, 300));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadAll(); }, [postId]);

  const minted = !!(detail?.meta?.minted);
  const mint = detail?.meta?.mint || {};
  const tokenId = mint?.token_id;
  const tx = mint?.tx_hash;
  const explorer = mint?.explorer;

  const sPre = Number(detail?.scores?.S_pre ?? 0);
  const cmtBonus = Number(detail?.meta?.score_extras?.comment_bonus ?? detail?.scores?.comment_bonus ?? 0);
  const sEff = Number(detail?.scores?.S_effective ?? sPre + cmtBonus);

  async function doLike() {
    try {
      setBusyLike(true);
      const out = await apiPost(`/posts/${postId}/like`, { to_address: addr || undefined });
      const msg = out.minted
        ? `좋아요 +1 (총 ${out.likes})\n공감 토큰 민팅됨\n• tokenId=${out.token_id}\n• tx=${String(out.tx_hash || '').slice(0, 24)}…`
        : `좋아요 +1 (총 ${out.likes})`;
      Alert.alert('좋아요', msg);
      await loadAll();
    } catch (e) {
      Alert.alert('실패', String(e).slice(0, 300));
    } finally {
      setBusyLike(false);
    }
  }

  async function addComment() {
    if (!cmt.trim()) return;
    try {
      setBusyCmt(true);
      await apiPost(`/posts/${postId}/comments`, { text: cmt, author: 'anon' });
      setCmt('');
      await loadAll(); // 댓글 보너스가 S_eff에 반영되도록 상세 재로딩
    } catch (e) {
      Alert.alert('실패', String(e).slice(0, 300));
    } finally {
      setBusyCmt(false);
    }
  }

  if (loading || !detail) {
    return (
      <View style={{ flex: 1, paddingTop: 40, paddingHorizontal: 12 }}>
        <Button title="← 목록" onPress={onBack} />
        <ActivityIndicator style={{ marginTop: 20 }} />
      </View>
    );
  }

  const likes = Number(detail?.meta?.likes ?? 0);

  return (
    <View style={{ flex: 1, paddingTop: 40, paddingHorizontal: 12 }}>
      <View style={s.headRow}>
        <Button title="← 목록" onPress={onBack} />
        <Text style={s.api}>API: {API}</Text>
      </View>

      <ScrollView style={{ paddingVertical: 10 }}>
        <Text style={s.title}>#{detail.id} · {detail.title || '(제목 없음)'}</Text>
        {!!detail.content && <Text style={{ marginTop: 6 }}>{detail.content}</Text>}

        <View style={{ height: 12 }} />
        <Text style={s.meta}>
          S_pre={sPre.toFixed(3)} + comment_bonus={cmtBonus.toFixed(2)} → S_eff={sEff.toFixed(3)}
          {'  '}· gate={String(detail?.gate)} · pass={String(detail?.scores?.gate_pass)}
        </Text>
        <Text style={s.meta}>likes: {likes}</Text>

        <View style={{ height: 16 }} />
        {minted ? (
          <View style={s.mintBox}>
            <Text style={s.minted}>✅ (본문) 민팅됨</Text>
            <Text>tokenId: {String(tokenId ?? '-')}</Text>
            <Text>tx: {(tx || '').slice(0, 20)}{tx ? '…' : ''}</Text>
            {explorer ? (
              <Button title="Etherscan에서 보기" onPress={() => Linking.openURL(explorer)} />
            ) : (
              <Text style={s.note}>시뮬레이션 모드일 수 있어요(익스플로러 없음)</Text>
            )}
          </View>
        ) : (
          <View style={s.mintBox}><Text style={s.notMinted}>⏳ (본문) 미민팅</Text></View>
        )}

        <View style={{ height: 18 }} />
        <Text style={s.section}>공감(좋아요) & 공감 토큰</Text>
        <View style={{ gap: 8 }}>
          <TextInput
            value={addr}
            onChangeText={setAddr}
            placeholder="내 지갑주소(옵션, 미입력 시 서버 ENV 사용)"
            style={s.input}
            autoCapitalize="none"
          />
          {busyLike ? <ActivityIndicator /> : <Button title="좋아요(+공감 토큰)" onPress={doLike} />}
        </View>

        <View style={{ height: 20 }} />
        <Text style={s.section}>댓글</Text>
        <View style={s.cmtRow}>
          <TextInput value={cmt} onChangeText={setCmt} placeholder="댓글 입력…" style={s.input} />
          {busyCmt ? <ActivityIndicator /> : <Button title="등록" onPress={addComment} />}
        </View>

        {(comments || []).map(x => (
          <View key={x.id} style={s.cmt}>
            <Text style={{ fontWeight: '600' }}>{x.author || 'anon'}</Text>
            <Text>{x.text}</Text>
            <Text style={{ color: '#888', fontSize: 12 }}>{x.created_at}</Text>
          </View>
        ))}

        <View style={{ height: 60 }} />
      </ScrollView>
    </View>
  );
}

// ───────────────────────── main ─────────────────────────
export default function CommunityApp() {
  const [items, setItems] = useState([]);
  const [selId, setSelId] = useState(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const out = await apiGet('/posts?limit=50&offset=0');
      setItems(out?.items || []);
    } catch (e) {
      Alert.alert('목록 실패', String(e).slice(0, 300));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  if (selId) return <PostDetail postId={selId} onBack={() => { setSelId(null); load(); }} />;

  return (
    <View style={{ flex: 1, paddingTop: 40, paddingHorizontal: 12 }}>
      <Text style={s.header}>커뮤니티</Text>
      <Text style={s.api}>API: {API}</Text>
      <View style={{ height: 8 }} />
      <Button title="새로고침" onPress={load} />
      <View style={{ height: 10 }} />
      {loading ? (
        <ActivityIndicator />
      ) : (
        <ScrollView>
          {items.map(p => (
            <PostCard key={String(p.id)} p={p} onOpen={() => setSelId(p.id)} />
          ))}
          <View style={{ height: 60 }} />
        </ScrollView>
      )}
    </View>
  );
}

// ───────────────────────── styles ─────────────────────────
const s = StyleSheet.create({
  header: { fontSize: 20, fontWeight: '800' },
  api: { color: '#666', fontSize: 12, marginLeft: 10 },
  card: { borderWidth: 1, borderColor: '#ddd', borderRadius: 12, padding: 12, marginBottom: 10, backgroundColor: '#fff' },
  title: { fontSize: 16, fontWeight: '700' },
  time: { fontSize: 12, color: '#666', marginLeft: 8 },
  meta: { color: '#444', marginTop: 6 },
  hint: { color: '#888', fontSize: 12, marginTop: 4 },
  minted: { color: '#0a7', marginTop: 4, fontWeight: '700' },
  notMinted: { color: '#a70', marginTop: 4, fontWeight: '700' },
  mintBox: { borderWidth: 1, borderColor: '#eee', borderRadius: 12, padding: 12, backgroundColor: '#fafafa' },
  note: { color: '#777', fontSize: 12, marginTop: 4 },
  headRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 6 },
  section: { fontWeight: '800', fontSize: 16, marginBottom: 6 },
  cmtRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 6 },
  input: { flex: 1, borderWidth: 1, borderColor: '#ddd', borderRadius: 8, padding: 8, backgroundColor: '#fff' },
  cmt: { marginTop: 8, padding: 10, borderRadius: 8, backgroundColor: '#fafafa', borderWidth: 1, borderColor: '#eee' }
});
