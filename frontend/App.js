import React, { useState, useEffect } from 'react';
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
  const [backendURL, setBackendURL] = useState('');

  useEffect(() => {
  const fetchNgrokURL = async () => {
    if (Platform.OS === 'ios' || Platform.OS === 'android') {
      try {
        const localIP = '172.30.1.66'; // 너의 백엔드가 실행 중인 IP
        const url = `http://${localIP}:8000/ngrok-url`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.ngrok_url) {
          console.log("✅ ngrok 주소 받아옴:", data.ngrok_url); // 👈 여기 추가
          setBackendURL(data.ngrok_url);
        } else {
          console.warn("⚠️ ngrok 주소 못 받아 fallback 사용");
          setBackendURL(`http://${localIP}:8000`);
        }
      } catch (error) {
        console.warn('❌ ngrok 주소 요청 실패:', error);
        setBackendURL(`http://${localIP}:8000`);
      }
    } else {
      setBackendURL('http://localhost:8000');
    }
  };

  fetchNgrokURL();
}, []);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('title', title);
    formData.append('content', content);

    try {
      const response = await fetch(`${backendURL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('요청 실패:', error);
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

      <Button
        title={loading ? '분석 중...' : '분석 요청'}
        onPress={handleSubmit}
        disabled={loading || !backendURL}
      />

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
          {result.raw_response && (
            <Text style={{ marginTop: 10 }}>{result.raw_response}</Text>
          )}
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
