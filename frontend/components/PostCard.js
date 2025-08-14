// components/PostCard.js
import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '../styles';

export default function PostCard({ p }) {
  const minted = p.minted ?? p.meta?.minted ?? false;
  const tokenId = p.token_id ?? p.meta?.mint?.token_id;

  return (
    <View style={styles.card}>
      <Text style={styles.title}>#{p.id} · {p.title || '(제목 없음)'}</Text>
      <Text style={styles.meta}>Gate: {p.gate ?? '-'} · 통과: {String(p.gate_pass)} · 좋아요: {p.likes ?? 0}</Text>
      <Text style={styles.preview}>{p.content?.slice(0, 100) ?? ''}</Text>
      <Text style={minted ? styles.minted : styles.notMinted}>
        {minted ? `✅ minted (token_id: ${tokenId ?? '-'})` : '⏳ not minted'}
      </Text>
    </View>
  );
}
