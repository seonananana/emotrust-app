import React, { useState } from 'react';
import {
  View,
  TextInput,
  Button,
  Text,
  StyleSheet,
  ScrollView,
  Platform,
} from 'react-native';

export default function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // ✅ 주소 자동 분기
  const isMobile = Platform.OS === 'ios' || Platform.OS === 'android';
  const backendBaseURL = isMobile
    ? 'http://192.168.137.1:8000' // 스마트폰 (핫스팟 연결)=>놑북-폰
    : 'http://localhost:8000';   // 노트북에서 개발 중일 경우

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('title', title);
    formData.append('content', content);

    try {
      const response = await fetch(`${backendBaseURL}/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: '요청 실패' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.label}>제목</Text>
      <TextInput
        style={styles.input}
        value={title}
        onChangeText={setTitle}
        placeholder="제목을 입력하세요"
      />

      <Text style={styles.label}>내용</Text>
      <TextInput
        style={[styles.input, { height: 100 }]}
        value={content}
        onChangeText={setContent}
        placeholder="내용을 입력하세요"
        multiline
      />

      <Button title={loading ? '분석 중...' : '분석 요청'} onPress={handleSubmit} disabled={loading} />

      {result && result.emotion_score !== undefined && (
        <View style={styles.resultBox}>
          <Text>📊 분석 결과</Text>
          <Text>감정 점수: {result.emotion_score}</Text>
          <Text>진정성 점수: {result.truth_score}</Text>
        </View>
      )}

      {result?.error && (
        <View style={styles.resultBox}>
          <Text style={{ color: 'red' }}>{result.error}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    marginTop: 50,
  },
  label: {
    fontWeight: 'bold',
    marginTop: 20,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginTop: 5,
    borderRadius: 5,
  },
  resultBox: {
    marginTop: 30,
    padding: 15,
    backgroundColor: '#eee',
    borderRadius: 5,
  },
});
