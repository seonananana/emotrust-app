// screens/AnalyzeScreen.js
import React, { useState } from 'react';
import { View, TextInput, Button, Text, Alert, ScrollView } from 'react-native';
import { styles } from '../styles';
import { fetchAnalyzeAndMint } from '../utils/api';

export default function AnalyzeScreen({ switchTab }) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [gate] = useState(0.3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await fetchAnalyzeAndMint({ title, content, gate });
      setResult(res);
    } catch (err) {
      Alert.alert('에러 발생', err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>공감 기반 신용평가</Text>

      <Text style={styles.label}>제목</Text>
      <TextInput style={styles.input} value={title} onChangeText={setTitle} placeholder="제목을 입력하세요" />

      <Text style={styles.label}>내용</Text>
      <TextInput
        style={[styles.input, { height: 120 }]}
        value={content}
        onChangeText={setContent}
        placeholder="내용을 입력하세요"
        multiline
      />

      <View style={{ marginTop: 16 }}>
        <Button title={loading ? '분석 중…' : '분석 요청'} onPress={handleSubmit} disabled={!title || !content} />
      </View>

      {result && (
        <View style={styles.resultBox}>
          <Text>정확성: {result.S_acc}</Text>
          <Text>진정성: {result.S_sinc}</Text>
          <Text>최종 점수: {result.S_pre} / 기준: {result.gate_used}</Text>
          <Text>NFT 발행: {result.gate_pass ? '✅ 성공' : '❌ 실패'}</Text>
          {result.tx_hash && <Text>Tx: {result.tx_hash}</Text>}
        </View>
      )}

      <Button title="커뮤니티로 이동" onPress={switchTab} />
    </ScrollView>
  );
}
