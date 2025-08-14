// App.js
import React, { useState } from 'react';
import { SafeAreaView, ScrollView, Text, TextInput, View, Button, Alert } from 'react-native';
import axios from 'axios';

const BACKEND_URL = 'https://your-ngrok-url.ngrok.io';

export default function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [gate, setGate] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    if (!content.trim()) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append('title', title);
      form.append('content', content);
      form.append('denom_mode', 'all');
      form.append('w_acc', 0.5);
      form.append('w_sinc', 0.5);
      form.append('gate', gate);
      const res = await axios.post(`${BACKEND_URL}/analyze-and-mint`, form);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      Alert.alert('분석 실패', '서버에 문제가 있습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, padding: 20 }}>
      <ScrollView>
        <Text>제목</Text>
        <TextInput value={title} onChangeText={setTitle} style={{ borderWidth: 1, marginBottom: 12, padding: 8 }} />

        <Text>내용</Text>
        <TextInput
          value={content}
          onChangeText={setContent}
          multiline
          style={{ borderWidth: 1, height: 120, padding: 8, marginBottom: 12 }}
        />

        <Button title={loading ? '분석 중...' : '분석 요청'} onPress={handleSubmit} disabled={loading} />

        {result && (
          <View style={{ marginTop: 20 }}>
            <Text>정확성: {result.S_acc}</Text>
            <Text>진정성: {result.S_sinc}</Text>
            <Text>최종 점수: {result.S_pre}</Text>
            <Text>Gate 통과 여부: {String(result.gate_pass)}</Text>
            {result.token_id && <Text>📦 NFT Token ID: {result.token_id}</Text>}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
