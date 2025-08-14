// components/PostCard.js
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function PostCard({ p }) {
  return (
    <View style={styles.card}>
      <Text style={styles.title}>#{p.id} · {p.title || '(제목 없음)'}</Text>
      <Text>Gate: {p.gate} · 통과: {String(p.gate_pass)} · 좋아요: {p.likes}</Text>
      <Text>{p.content?.slice(0, 100)}...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { padding: 12, borderBottomWidth: 1 },
  title: { fontWeight: 'bold' },
});
