// frontend/CommunityApp.js
import React, { useEffect, useState } from 'react';
import { View, Text, Button, ScrollView, ActivityIndicator, Alert, Linking, TextInput, StyleSheet, TouchableOpacity } from 'react-native';

const API = (process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/,'');

async function apiGet(path){ const r=await fetch(`${API}${path}`); if(!r.ok) throw new Error(await r.text()); return r.json(); }
async function apiPost(path, body){ const r=await fetch(`${API}${path}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}); if(!r.ok) throw new Error(await r.text()); return r.json(); }

function PostCard({ p, onOpen }) {
  return (
    <TouchableOpacity style={s.card} onPress={onOpen}>
      <Text style={s.title}>#{p.id} · {p.title || '(제목 없음)'}</Text>
      <Text style={s.meta}>
        Gate: {p.gate ?? '-'} · 통과: {String(p.gate_pass)} · 좋아요: {p.likes ?? 0}
      </Text>
      {p.minted ? (
        <Text style={s.minted}>✅ minted (token_id: {p.token_id ?? '-'})</Text>
      ) : (
        <Text style={s.notMinted}>⏳ not minted</Text>
      )}
    </TouchableOpacity>
  );
}

function PostDetail({ post, onBack }) {
  const [busy, setBusy] = useState(false);
  const [mint, setMint] = useState(null);
  const [comments, setComments] = useState([]);
  const [cmt, setCmt] = useState('');

  const loadComments = async () => {
    try { const out = await apiGet(`/posts/${post.id}/comments`); setComments(out.items || []); }
    catch(e){ console.error(e); }
  };

  useEffect(()=>{ loadComments(); }, [post?.id]);

  const doLike = async () => {
    try { const out = await apiPost(`/posts/${post.id}/like`, {}); Alert.alert('좋아요', `+1 (현재 ${out.likes})`); }
    catch(e){ Alert.alert('실패', String(e).slice(0,300)); }
  };

  const addComment = async () => {
    if (!cmt.trim()) return;
    try {
      await apiPost(`/posts/${post.id}/comments`, { text: cmt });
      setCmt(''); loadComments();
    } catch(e){ Alert.alert('실패', String(e).slice(0,300)); }
  };

  const mintNow = async () => {
    try {
      setBusy(true); setMint(null);
      const text = `${post.title ? post.title + '\n\n' : ''}${post.content || ''}`;
      const out = await apiPost(`/analyze-mint`, { text, denom_mode: post.denom_mode || 'all' });
      setMint(out);
      if (out.minted && out.token_id != null) {
        // post_id ↔ token 저장
        await apiPost(`/posts/${post.id}/set-mint`, { token_id: out.token_id, tx_hash: out.tx_hash, explorer: out.explorer });
      }
      if (out.minted) {
        Alert.alert('민팅 완료', `token_id: ${out.token_id}`);
      } else {
        Alert.alert('게이트 미통과', '점수만 계산되었습니다.');
      }
    } catch(e){ Alert.alert('민팅 실패', String(e).slice(0,300)); }
    finally { setBusy(false); }
  };

  return (
    <View style={{flex:1}}>
      <View style={s.headRow}>
        <Button title="← 목록" onPress={onBack} />
        <Text style={s.api}>API: {API}</Text>
      </View>
      <ScrollView style={{padding:12}}>
        <Text style={s.title}>#{post.id} · {post.title || '(제목 없음)'}</Text>
        <Text style={{marginTop:6}}>{post.content}</Text>
        <View style={{height:10}} />
        <Text style={s.meta}>
          Gate: {post.gate ?? '-'} · 통과: {String(post.scores?.gate_pass ?? post.gate_pass)}
        </Text>
        <View style={{height:10}} />

        <View style={s.row}>
          <Button title="좋아요" onPress={doLike} />
          <View style={{width:10}} />
          {busy ? <ActivityIndicator /> : <Button title="민팅하기" onPress={mintNow} />}
        </View>

        {mint && (
          <View style={s.mintBox}>
            <Text style={s.mintTitle}>민팅 결과</Text>
            <Text>minted: {String(mint.minted)}</Text>
            {mint.scores && <Text>S_pre: {mint.scores.S_pre} · S_sinc: {mint.scores.S_sinc}</Text>}
            {mint.token_id != null && <Text>token_id: {mint.token_id}</Text>}
            {mint.tx_hash && <Text numberOfLines={1}>tx: {mint.tx_hash}</Text>}
            {mint.explorer && <Button title="Etherscan에서 보기" onPress={()=>Linking.openURL(mint.explorer)} />}
          </View>
        )}

        <View style={{height:20}} />
        <Text style={s.section}>댓글</Text>
        <View style={s.cmtRow}>
          <TextInput value={cmt} onChangeText={setCmt} placeholder="댓글 입력…" style={s.input} />
          <Button title="등록" onPress={addComment} />
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

export default function CommunityApp(){
  const [items, setItems] = useState([]);
  const [sel, setSel] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      setLoading(true);
      const out = await apiGet('/posts?limit=50&offset=0');
      setItems(out.items || []);
    } catch(e){ Alert.alert('목록 실패', String(e).slice(0,300)); }
    finally { setLoading(false); }
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
          {items.map(p => <PostCard key={p.id} p={p} onOpen={()=>setSel(p)} />)}
          <View style={{height:60}} />
        </ScrollView>
      )}
    </View>
  );
}

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
  mintBox:{marginTop:12,padding:12,backgroundColor:'#f4f7ff',borderRadius:10},
  mintTitle:{fontWeight:'800',marginBottom:6},
  section:{fontWeight:'800',fontSize:16},
  cmtRow:{flexDirection:'row',alignItems:'center',gap:8,marginTop:6},
  input:{flex:1,borderWidth:1,borderColor:'#ddd',borderRadius:8,padding:8,backgroundColor:'#fff'},
  cmt:{marginTop:8,padding:10,borderRadius:8,backgroundColor:'#fafafa',borderWidth:1,borderColor:'#eee'}
});
