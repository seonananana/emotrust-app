// CommunityApp.js
import React, { useEffect, useState } from 'react';
import {
  SafeAreaView,
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
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ───────────────────────── UI: Card ─────────────────────────
function PostCard({ p, onOpen }) {
  const minted = p.minted ?? p.meta?.minted ?? false;
  const tokenId = p.token_id ?? p.meta?.mint?.token_id;

  return (
    <TouchableOpacity style={s.card} onPress={onOpen} activeOpacity={0.85}>
      <Text style={s.title}>#{p.id} · {p.title || '(제목 없음)'}</Text>

      {/* ✅ 내용(본문) 추가 */}
      {p.content && (
        <Text style={s.content} numberOfLines={2}>
          {p.content}
        </Text>
      )}

      <Text style={s.meta}>
        Gate: {p.gate ?? '-'} · 통과: {String(p.gate_pass ?? p.scores?.gate_pass ?? false)} · 좋아요: {p.likes ?? p.meta?.likes ?? 0}
      </Text>

      {minted ? (
        <Text style={s.minted}>✅ minted (token_id: {tokenId ?? '-'})</Text>
      ) : (
        <Text style={s.notMinted}>⏳ not minted</Text>
      )}
    </TouchableOpacity>
  );
}

// ─────────────────────── UI: Detail View ───────────────────────
function PostDetail({ post, onBack }) {
  const [comments, setComments] = useState([]);
  const [cmt, setCmt] = useState('');
  const [busy, setBusy] = useState(false);

  const minted = !!(post.minted || post.meta?.minted);
  const explorer = post.explorer || post.meta?.mint?.explorer;
  const tokenId = post.token_id ?? post.meta?.mint?.token_id;

  const loadComments = async () => {
    try {
      const out = await apiGet(`/posts/${post.id}/comments`);
      setComments(out.items || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => { loadComments(); }, [post?.id]);

  const doLike = async () => {
    try {
      const out = await apiPost(`/posts/${post.id}/like`, {});
      Alert.alert('좋아요', `+1 (현재 ${out.likes})`);
    } catch (e) {
      Alert.alert('실패', String(e).slice(0, 300));
    }
  };

  const addComment = async () => {
    if (!cmt.trim()) return;
    try {
      setBusy(true);
      await apiPost(`/posts/${post.id}/comments`, { text: cmt });
      setCmt('');
      await loadComments();
    } catch (e) {
      Alert.alert('실패', String(e).slice(0, 300));
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={{ flex: 1 }}>
      {/* 상세 내부 상단바(목록으로) */}
      <View style={s.topbar}>
        <TouchableOpacity onPress={onBack} style={s.topBtn}>
          <Text style={s.topBtnTx}>← 목록</Text>
        </TouchableOpacity>
        <Text style={s.topTitle} numberOfLines={1}>게시글</Text>
        <View style={{ width: 72 }} />
      </View>

      <ScrollView style={{ paddingHorizontal: 12 }} contentContainerStyle={{ paddingBottom: 80 }}>
        <Text style={[s.title, { marginTop: 12 }]}>#{post.id} · {post.title || '(제목 없음)'}</Text>
        {!!post.content && <Text style={{ marginTop: 6 }}>{post.content}</Text>}

        <View style={{ height: 10 }} />
        <Text style={s.meta}>
          Gate: {post.gate ?? '-'} · 통과: {String(post.scores?.gate_pass ?? post.gate_pass)}
        </Text>

        <View style={{ height: 12 }} />
        <View style={s.row}>
          <Button title="좋아요" onPress={doLike} />
          <View style={{ width: 10 }} />
          {minted ? (
            explorer
              ? <Button title="Etherscan" onPress={() => Linking.openURL(explorer)} />
              : <Text>민팅 완료 (token_id: {tokenId ?? '-'})</Text>
          ) : (
            <Text style={{ color: '#a70' }}>자동 민팅 대기/실패</Text>
          )}
        </View>

        <View style={{ height: 20 }} />
        <Text style={s.section}>댓글</Text>
        <View style={s.cmtRow}>
          <TextInput value={cmt} onChangeText={setCmt} placeholder="댓글 입력…" style={s.input} />
          {busy ? <ActivityIndicator /> : <Button title="등록" onPress={addComment} />}
        </View>

        {comments.map(x => (
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
export default function CommunityApp({ onBackToAnalyze }) {
  const [items, setItems] = useState([]);
  const [sel, setSel] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      setLoading(true);
      const out = await apiGet('/posts?limit=50&offset=0');
      const items = (out.items || []).map(x => {
        const minted = x.minted ?? x.meta?.minted ?? false;
        const token_id = x.token_id ?? x.meta?.mint?.token_id;
        const explorer = x.explorer ?? x.meta?.mint?.explorer;
        const likes = x.likes ?? x.meta?.likes ?? 0;
        return { ...x, minted, token_id, explorer, likes };
      });
      setItems(items);
    } catch (e) {
      Alert.alert('목록 실패', String(e).slice(0, 300));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (sel) return <PostDetail post={sel} onBack={() => setSel(null)} />;

  return (
    <SafeAreaView style={{ flex: 1 }}>
      {/* ⬇️ 고정 상단바 (좌: 작성으로, 중: 커뮤니티, 우: 새로고침) */}
      <View style={s.topbar}>
        {onBackToAnalyze ? (
          <TouchableOpacity onPress={onBackToAnalyze} style={s.topBtn}>
            <Text style={s.topBtnTx}>← 작성으로</Text>
          </TouchableOpacity>
        ) : <View style={{ width: 72 }} />}

        <Text style={s.topTitle}>커뮤니티</Text>

        <TouchableOpacity onPress={load} style={s.topBtn}>
          <Text style={s.topBtnTx}>새로고침</Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          <ActivityIndicator />
        </View>
      ) : (
        <ScrollView style={{ paddingHorizontal: 12 }} contentContainerStyle={{ paddingTop: 8, paddingBottom: 80 }}>
          {items.map(p => (
            <PostCard key={`${p.id}-${p.updated_at || ''}`} p={p} onOpen={() => setSel(p)} />
          ))}
          <View style={{ height: 60 }} />
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

// ───────────────────────── styles ─────────────────────────
const s = StyleSheet.create({
  topbar: {
    height: 48,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderColor: '#eee',
    backgroundColor: '#fff',
  },
  topBtn: { paddingVertical: 6, paddingHorizontal: 8, borderRadius: 8 },
  topBtnTx: { fontWeight: '700', color: '#111' },
  topTitle: { fontSize: 16, fontWeight: '800' },

  card: { borderWidth: 1, borderColor: '#ddd', borderRadius: 12, padding: 12, marginTop: 10, backgroundColor: '#fff' },
  title: { fontSize: 16, fontWeight: '700' },
  meta: { color: '#666', marginTop: 4 },
  minted: { color: '#0a7', marginTop: 6, fontWeight: '700' },
  notMinted: { color: '#a70', marginTop: 6, fontWeight: '700' },
  row: { flexDirection: 'row', alignItems: 'center' },

  section: { fontWeight: '800', fontSize: 16, marginBottom: 6 },
  cmtRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 6 },
  input: { flex: 1, borderWidth: 1, borderColor: '#ddd', borderRadius: 8, padding: 8, backgroundColor: '#fff' },
  cmt: { marginTop: 8, padding: 10, borderRadius: 8, backgroundColor: '#fafafa', borderWidth: 1, borderColor: '#eee' }
});
