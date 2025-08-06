import React, { useState } from 'react';
import { View, TextInput, Button, Text, StyleSheet, ScrollView } from 'react-native';

export default function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

const handleSubmit = async () => {
  setLoading(true);
  setResult(null);

  const formData = new FormData();
  formData.append('title', title);
  formData.append('content', content);

  try {
    const response = await fetch('http://172.20.10.2:8000/analyze', {
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

      <Button title="분석 요청" onPress={handleSubmit} />

      {result && result.emotion_score !== undefined && (
  <View style={styles.resultBox}>
    <Text>📊 분석 결과</Text>
    <Text>감정 점수: {result.emotion_score}</Text>
    <Text>진정성 점수: {result.truth_score}</Text>
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
