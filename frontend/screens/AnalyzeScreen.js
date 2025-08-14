// screens/AnalyzeScreen.js
import React, { useState } from 'react';
import { View, TextInput, Button, Text, ScrollView } from 'react-native';
import { fetchJSON } from '../utils/api';
import ScoreGauge from '../components/ScoreGauge';

export default function AnalyzeScreen() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const res = await fetchJSON('/analyze', {
      title,
      content,
      w_acc: 0.5,
      w_sinc: 0.5,
      gate: 0.3,
    });
    setResult(res);
  };

  return (
    <ScrollView contentContainerStyle={{ padding: 20 }}>
      <Text>제목</Text>
      <TextInput value={title} onChangeText={setTitle} style={{ borderWidth: 1 }} />
      <Text>내용</Text>
      <TextInput value={content} onChangeText={setContent} style={{ borderWidth: 1, height: 100 }} multiline />
      <Button title="분석 요청" onPress={handleSubmit} />
      {result && (
        <>
          <ScoreGauge label="정확성" score={result.S_acc} />
          <ScoreGauge label="진정성" score={result.S_sinc} />
          <Text>통합 점수: {result.S_pre.toFixed(3)}</Text>
          <Text>통과 여부: {result.gate_pass ? '✅ 통과' : '❌ 불가'}</Text>
        </>
      )}
    </ScrollView>
  );
}
