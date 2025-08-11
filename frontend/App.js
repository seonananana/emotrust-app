// frontend/App.js
import React, { useState, useEffect, useMemo } from 'react';
import {
  View,
  TextInput,
  Button,
  Text,
  StyleSheet,
  ScrollView,
  Platform,
  ActivityIndicator,
  KeyboardAvoidingView,
  Alert,
  TouchableOpacity,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';

// ====== ENV (ngrok HTTPS만 허용) ======
const RAW_ENV_URL = process.env.EXPO_PUBLIC_API_BASE_URL;
const normalizeUrl = (u) => (u || '').trim().replace(/\/+$/, '');

// fetch JSON with timeout
async function fetchJSON(url, { method = 'GET', headers, body, timeout = 5000 } = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, { method, headers, body, signal: controller.signal });
    const text = await res.text();
    let data = null;
    try { data = JSON.parse(text); } catch {}
    return { ok: res.ok, status: res.status, data, raw: text };
  } finally {
    clearTimeout(id);
  }
}

export default function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);

  const [backendURL, setBackendURL] = useState('');
  const [backendSource, setBackendSource] = useState(''); // 'env'
  const [bootstrapping, setBootstrapping] = useState(true);
  const [loading, setLoading] = useState(false);

  // 파일 업로드(PDF) 상태
  const [pdfs, setPdfs] = useState([]); // [{ uri, name, mimeType, size }...]

  // ✅ 초기 Base URL: env(HTTPS)만 허용 — 폴백 없음
  useEffect(() => {
    if (__DEV__) {
      console.log('ENV EXPO_PUBLIC_API_BASE_URL =', RAW_ENV_URL);
    }
    const init = async () => {
      const envUrl = normalizeUrl(RAW_ENV_URL);

      if (!envUrl) {
        Alert.alert(
          'API 주소 미설정',
          'frontend/.env 파일에\nEXPO_PUBLIC_API_BASE_URL=https://<ngrok>.ngrok-free.app\n를 설정하세요.'
        );
        setBackendURL('');
        setBackendSource('');
        setBootstrapping(false);
        return;
      }
      if (!envUrl.startsWith('https://')) {
        Alert.alert('HTTPS만 허용', `현재 값: ${envUrl}`);
        setBackendURL('');
        setBackendSource('');
        setBootstrapping(false);
        return;
      }

      setBackendURL(envUrl);
      setBackendSource('env');
      setBootstrapping(false);
    };
    init();
  }, []);

  const canSubmit = useMemo(() => {
    return !!backendURL && !loading && title.trim().length > 0 && content.trim().length > 0;
  }, [backendURL, loading, title, content]);

  const pickPDFs = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: 'application/pdf',
        multiple: true,                  // 여러 파일 선택
        copyToCacheDirectory: true,
      });
      if (res.canceled) return;

      // SDK별로 assets 또는 output 배열 형태가 다를 수 있음 → 통일 처리
      const assets = res.assets || [];
      const next = [...pdfs];

      assets.forEach((a) => {
        if (!a?.uri) return;
        // 중복 방지(같은 uri면 스킵)
        if (next.find(x => x.uri === a.uri)) return;
        next.push({
          uri: a.uri,
          name: a.name || `evidence_${Date.now()}.pdf`,
          mimeType: a.mimeType || 'application/pdf',
          size: a.size ?? 0,
        });
      });

      setPdfs(next);
    } catch (e) {
      Alert.alert('파일 선택 오류', String(e));
    }
  };

  const removePDF = (idx) => {
    setPdfs((prev) => prev.filter((_, i) => i !== idx));
  };

  const clearPDFs = () => setPdfs([]);

  const handleSubmit = async () => {
    if (!backendURL) {
      Alert.alert('백엔드 주소 없음', 'EXPO_PUBLIC_API_BASE_URL(HTTPS)을 설정하세요.');
      return;
    }
    if (!title.trim() || !content.trim()) {
      Alert.alert('입력 필요', '제목과 내용을 모두 입력해주세요.');
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('title', title.trim());
    formData.append('content', content.trim());
    // 선택: 가중치/모드/게이트 기본값은 서버 기본과 동일 (원하면 UI로 노출 가능)
    formData.append('denom_mode', 'all'); // or 'matched'
    formData.append('w_acc', String(0.5));
    formData.append('w_sinc', String(0.5));
    formData.append('gate', String(0.70));

    // PDF 첨부 (여러 개는 같은 키 'pdfs'로 반복 append)
    for (const f of pdfs) {
      formData.append('pdfs', {
        uri: f.uri,
        name: f.name || 'evidence.pdf',
        type: f.mimeType || 'application/pdf',
      });
    }

    try {
      const { ok, status, data, raw } = await fetchJSON(`${backendURL}/analyze`, {
        method: 'POST',
        body: formData,
        timeout: 25000,
      });

      if (!ok) {
        setResult({
          error: `서버 오류 (HTTP ${status})`,
          raw_response: typeof raw === 'string' ? raw.slice(0, 1000) : JSON.stringify(data)?.slice(0, 1000),
        });
        return;
      }

      // analyzer 기반 main.py의 응답 스키마에 맞게 처리
      if (!data?.ok || !data?.result) {
        setResult({
          error: '응답 형식이 올바르지 않습니다.',
          raw_response: JSON.stringify(data)?.slice(0, 1000),
        });
        return;
      }

      setResult(data); // { ok, meta, result:{...} } 형태 그대로 저장
    } catch (error) {
      setResult({ error: `요청 실패: ${String(error)}` });
    } finally {
      setLoading(false);
    }
  };

  const filesInfo = useMemo(() => {
    const count = pdfs.length;
    const totalBytes = pdfs.reduce((acc, f) => acc + (f.size || 0), 0);
    return {
      count,
      sizeLabel:
        totalBytes > 0
          ? (totalBytes / (1024 * 1024)).toFixed(2) + ' MB'
          : count > 0 ? '크기 미상' : '0',
    };
  }, [pdfs]);

  return (
    <KeyboardAvoidingView
      behavior={Platform.select({ ios: 'padding', android: undefined })}
      style={{ flex: 1 }}
    >
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        {/* 상태/디버그 박스 */}
        <View style={styles.debugBox}>
          <Text style={styles.debugTitle}>Backend</Text>
          <Text selectable style={styles.debugText}>
            URL: {backendURL || '(없음)'}
          </Text>
          <Text style={styles.debugText}>Source: {backendSource || '-'}</Text>
          <Text style={[styles.debugText, { marginTop: 4 }]}>
            ENV: {normalizeUrl(RAW_ENV_URL) || '(미설정)'}
          </Text>
          {bootstrapping && (
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
              <ActivityIndicator />
              <Text style={styles.debugText}>주소 확인 중…</Text>
            </View>
          )}
          <Text style={[styles.debugText, { marginTop: 6 }]}>
            📎 PDFs: {filesInfo.count}개 ({filesInfo.sizeLabel})
          </Text>
        </View>

        <Text style={styles.label}>제목</Text>
        <TextInput
          style={styles.input}
          value={title}
          onChangeText={setTitle}
          placeholder="제목을 입력하세요"
          autoCapitalize="none"
          autoCorrect={false}
        />

        <Text style={styles.label}>내용</Text>
        <TextInput
          style={[styles.input, { height: 120 }]}
          value={content}
          onChangeText={setContent}
          placeholder="내용을 입력하세요"
          multiline
        />

        {/* 파일 업로드 UI */}
        <View style={{ marginTop: 12, gap: 8 }}>
          <Button title="📎 PDF 첨부" onPress={pickPDFs} />
          {pdfs.length > 0 && (
            <View style={styles.filesBox}>
              {pdfs.map((f, i) => (
                <View key={f.uri + i} style={styles.fileRow}>
                  <Text numberOfLines={1} style={{ flex: 1 }}>
                    {f.name || 'evidence.pdf'}
                  </Text>
                  <TouchableOpacity onPress={() => removePDF(i)} style={styles.removeBtn}>
                    <Text style={{ color: '#b00020', fontWeight: '700' }}>삭제</Text>
                  </TouchableOpacity>
                </View>
              ))}
              <View style={{ marginTop: 6 }}>
                <Button title="첨부 초기화" color="#64748b" onPress={clearPDFs} />
              </View>
            </View>
          )}
        </View>

        <View style={{ marginTop: 16 }}>
          <Button
            title={loading ? '분석 중…' : '분석 요청'}
            onPress={handleSubmit}
            disabled={!canSubmit || bootstrapping}
          />
        </View>

        {/* 결과 표시 (analyzer 기반) */}
        {result?.result && (
          <View style={styles.resultBox}>
            <Text style={styles.resultTitle}>📊 분석 결과</Text>
            <Text>최종 점수(S_pre): {(result.result.S_pre * 100).toFixed(1)}</Text>
            <Text>진정성(S_sinc): {(result.result.S_sinc * 100).toFixed(1)}</Text>
            <Text>
              팩트(S_fact): {result.result.S_fact == null ? '검증 불가' : (result.result.S_fact * 100).toFixed(1)}
            </Text>
            <Text>커버리지: {(result.result.coverage * 100).toFixed(1)}%</Text>
            <Text>토큰 수: {result.result.total} / 매칭: {result.result.matched}</Text>
            <Text>PII 처리: {result.result.masked ? '마스킹됨' : '그대로'}</Text>
            <Text>게이트 통과: {result.result.gate_pass ? '✅' : '❌'}</Text>
            {!!(result?.meta) && (
              <Text style={{ marginTop: 6, opacity: 0.7 }}>
                제목 길이: {result.meta.title?.length || 0} / 본문 길이: {result.meta.chars}
              </Text>
            )}
          </View>
        )}

        {/* 에러 박스 */}
        {result?.error && (
          <View style={[styles.resultBox, { backgroundColor: '#ffe6e6', borderColor: '#ffcccc' }]}>
            <Text style={{ color: '#b00020', fontWeight: '600' }}>{result.error}</Text>
            {result.raw_response && (
              <Text style={{ marginTop: 8, color: '#333' }}>{result.raw_response}</Text>
            )}
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 6 },
  label: { fontWeight: '600', marginTop: 14, marginBottom: 6 },
  input: {
    borderWidth: 1, borderColor: '#d4d4d8', padding: 10, borderRadius: 8, backgroundColor: '#fff',
  },
  filesBox: {
    marginTop: 6, padding: 10, backgroundColor: '#f8fafc',
    borderWidth: 1, borderColor: '#e2e8f0', borderRadius: 8, gap: 6,
  },
  fileRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  removeBtn: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6, borderWidth: 1, borderColor: '#fecaca' },
  resultBox: {
    marginTop: 24, padding: 16, backgroundColor: '#eef2ff',
    borderRadius: 8, borderWidth: 1, borderColor: '#c7d2fe', gap: 4,
  },
  resultTitle: { fontWeight: '700', marginBottom: 6 },
  debugBox: {
    marginBottom: 10, padding: 12, backgroundColor: '#f8fafc',
    borderWidth: 1, borderColor: '#e2e8f0', borderRadius: 8,
  },
  debugTitle: { fontWeight: '700', marginBottom: 6 },
  debugText: { color: '#334155' },
});
