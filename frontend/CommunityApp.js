// CommunityApp.js
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

const API = (process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/,'');

// ───────────────────────── helpers ─────────────────────────
async function apiGet(path){
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiPost(path, body){
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
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
    <TouchableOpacity style={s.card} onPress={onOpen}>
      <Text style={s.title}>#{p.id} · {p.title || '(제목 없음)'}</Text>
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
      Alert.alert('실패', String(e).slice(0,300));
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
      Alert.alert('실패', String(e).slice(0,300));
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={{flex:1}}>
      <View style={s.headRow}>
        <Button title="← 목록" onPress={onBack} />
        <Text style={s.api}>API: {API}</Text>
      </View>
      <ScrollView style={{padding:12}}>
        <Text style={s.title}>#{post.id} · {post.title || '(제목 없음)'}</Text>
        {!!post.content && <Text style={{marginTop:6}}>{post.content}</Text>}

        <View style={{height:10}} />
        <Text style={s.meta}>
          Gate: {post.gate ?? '-'} · 통과: {String(post.scores?.gate_pass ?? post.gate_pass)}
        </Text>

        <View style={{height:12}} />
        <View style={s.row}>
          <Button title="좋아요" onPress={doLike} />
          <View style={{width:10}} />
          {minted ? (
            explorer
              ? <Button title="Etherscan" onPress={() => Linking.openURL(explorer)} />
              : <Text>민팅 완료 (token_id: {tokenId ?? '-'})</Text>
          ) : (
            <Text style={{color:'#a70'}}>자동 민팅 대기/실패</Text>
          )}
        </View>

        <View style={{height:20}} />
        <Text style={s.section}>댓글</Text>
        <View style={s.cmtRow}>
          <TextInput value={cmt} onChangeText={setCmt} placeholder="댓글 입력…" style={s.input} />
          {busy ? <ActivityIndicator /> : <Button title="등록" onPress={addComment} />}
        </View>

        {comments.map(x=>(
          <View key={x.id} style={s.cmt}>
            <Text style={{fontWeight:'600'}}>{x.author || 'anon'}</Text>
            <Text>{x.text}</Text>
            <Text style={{color:'#888', fontSize:12}}>{x.created_at}</Text>
          </View>
        ))}

        <View style={{height:60}} />
      </ScrollView>
    </View>
  );
}

// ───────────────────────── main ─────────────────────────
export default function CommunityApp(){
  const [items, setItems] = useState([]);
  const [sel, setSel] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      setLoading(true);
      const out = await apiGet('/posts?limit=50&offset=0');
      // minted/token_id/explorer 보정 (파일모드/DB모드 모두 대응)
      const items = (out.items || []).map(x => {
        const minted = x.minted ?? x.meta?.minted ?? false;
        const token_id = x.token_id ?? x.meta?.mint?.token_id;
        const explorer = x.explorer ?? x.meta?.mint?.explorer;
        const likes = x.likes ?? x.meta?.likes ?? 0;
        return { ...x, minted, token_id, explorer, likes };
      });
      setItems(items);
    } catch (e) {
      Alert.alert('목록 실패', String(e).slice(0,300));
    } finally {
      setLoading(false);
    }
  };

  useEffect(()=>{ load(); }, []);

  if (sel) return <PostDetail post={sel} onBack={()=>{ setSel(null); load(); }} />;

  return (
    <View style={{flex:1, paddingTop:40, paddingHorizontal:12}}>
      <Text style={s.header}>커뮤니티</Text>
      <Text style={s.api}>API: {API}</Text>
      <View style={{height:8}} />
      <Button title="새로고침" onPress={load} />
      <View style={{height:10}} />
      {loading ? <ActivityIndicator/> : (
        <ScrollView>
          {items.map(p => (
            <PostCard key={`${p.id}-${p.updated_at || ''}`} p={p} onOpen={()=>setSel(p)} />
          ))}
          <View style={{height:60}} />
        </ScrollView>
      )}
    </View>
  );
}

// ───────────────────────── styles ─────────────────────────
const s = StyleSheet.create({
  header:{fontSize:20,fontWeight:'800'},
  api:{color:'#666',fontSize:12,marginLeft:10},
  card:{borderWidth:1,borderColor:'#ddd',borderRadius:12,padding:12,marginBottom:10,backgroundColor:'#fff'},
  title:{fontSize:16,fontWeight:'700'},
  meta:{color:'#666',marginTop:4},
  minted:{color:'#0a7',marginTop:6,fontWeight:'700'},
  notMinted:{color:'#a70',marginTop:6,fontWeight:'700'},
  row:{flexDirection:'row',alignItems:'center'},
  headRow:{flexDirection:'row',alignItems:'center',justifyContent:'space-between',padding:8},
  section:{fontWeight:'800',fontSize:16, marginBottom:6},
  cmtRow:{flexDirection:'row',alignItems:'center',gap:8,marginTop:6},
  input:{flex:1,borderWidth:1,borderColor:'#ddd',borderRadius:8,padding:8,backgroundColor:'#fff'},
  cmt:{marginTop:8,padding:10,borderRadius:8,backgroundColor:'#fafafa',borderWidth:1,borderColor:'#eee'}
});
