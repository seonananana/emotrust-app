// App.js
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
  Linking,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import CommunityApp from './CommunityApp';
import { SafeAreaView } from 'react-native';

const API = (process.env.EXPO_PUBLIC_API_BASE_URL || '').replace(/\/+$/, '');
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
  // 화면 탭: 'analyze' | 'community'
  const [tab, setTab] = useState('analyze');

  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);

  const [backendURL, setBackendURL] = useState('');
  const [loading, setLoading] = useState(false);

  // 파일 업로드(PDF) 상태
  const [pdfs, setPdfs] = useState([]); // [{ uri, name, mimeType, size }...]

  // 저장 상태
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState(null);
  const [mintInfo, setMintInfo] = useState(null); // { minted, token_id, tx_hash, explorer, mint_error }

  // 게이트(정규화 0~1)
  const [gate, setGate] = useState(0.70);

  // ✅ 초기 Base URL: env(HTTPS)만 허용 — 폴백 없음
  useEffect(() => {
    const envUrl = normalizeUrl(RAW_ENV_URL);
    if (!envUrl) {
      Alert.alert(
        'API 주소 미설정',
        'frontend/.env 파일에\nEXPO_PUBLIC_API_BASE_URL=https://<ngrok>.ngrok-free.app\n를 설정하세요.'
      );
      return;
    }
    if (!envUrl.startsWith('https://')) {
      Alert.alert('HTTPS만 허용', `현재 값: ${envUrl}`);
      return;
    }
    setBackendURL(envUrl);
  }, []);

  const canSubmit = useMemo(() => {
    return !!backendURL && !loading && title.trim().length > 0 && content.trim().length > 0;
  }, [backendURL, loading, title, content]);

  const pickPDFs = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: 'application/pdf',
        multiple: true,
        copyToCacheDirectory: true,
      });
      if (res.canceled) return;

      const assets = res.assets || [];
      const next = [...pdfs];

      assets.forEach((a) => {
        if (!a?.uri) return;
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

  const removePDF = (idx) => setPdfs((prev) => prev.filter((_, i) => i !== idx));
  const clearPDFs = () => setPdfs([]);

  // ✅ 게이트 통과 시 저장(API: /posts) — 백엔드가 자동 민팅 수행
  const savePost = async ({ analysis, meta }) => {
    setSaving(true);
    setSavedId(null);
    setMintInfo(null);

    const payload = {
      title,
      content,
      // ⚠️ 서버로는 보내되, 화면에는 노출하지 않음(민감/내부용)
      scores: {
        S_pre: analysis.S_pre,
        S_sinc: analysis.S_sinc,
        S_fact: analysis.S_fact ?? null,
        coverage: analysis.coverage,
        total: analysis.total,
        matched: analysis.matched,
        masked: analysis.masked,
        gate_pass: analysis.gate_pass,
      },
      weights: { w_acc: 0.5, w_sinc: 0.5 },
      denom_mode: meta?.denom_mode || 'all',
      gate: meta?.gate ?? gate,
      files: pdfs.map(f => ({ name: f.name, size: f.size })),
      meta: {
        ...meta,
        title_len: title.length,
        content_len: content.length,
        masked_text: analysis.clean_text,
      },
      analysis_id: meta?.analysis_id || null,
    };

    const { ok, status, data, raw } = await fetchJSON(`${backendURL}/posts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      timeout: 20000,
    });

    setSaving(false);
    if (!ok) {
      Alert.alert('저장 실패', `HTTP ${status}\n${(data?.message || raw || '').slice(0, 200)}`);
      return;
    }

    const id = data?.post_id || data?.id;
    setSavedId(id || null);

    // 자동 민팅 결과 저장 + 알림
    const info = {
      minted: !!data?.minted,
      token_id: data?.token_id ?? null,
      tx_hash: data?.tx_hash ?? null,
      explorer: data?.explorer ?? null,
      mint_error: data?.mint_error ?? null,
    };
    setMintInfo(info);

    if (info.minted) {
      Alert.alert(
        '등록+민팅 완료',
        `#${id}\ntoken_id: ${info.token_id}\nTx: ${info.tx_hash}`,
        info.explorer
          ? [{ text: 'Etherscan', onPress: () => Linking.openURL(info.explorer) }, { text: '확인' }]
          : [{ text: '확인' }]
      );
    } else {
      const why = info.mint_error ? `\n(민팅 실패: ${String(info.mint_error).slice(0,140)})` : '';
      Alert.alert('등록 완료', `글이 저장되었습니다. #${id}${why}`);
    }
  };

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
    setSavedId(null);
    setMintInfo(null);

    const formData = new FormData();
    formData.append('title', title.trim());
    formData.append('content', content.trim());
    formData.append('denom_mode', 'all');
    formData.append('w_acc', String(0.5));
    formData.append('w_sinc', String(0.5));
    formData.append('gate', String(gate));

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

      if (!data?.ok || !data?.result) {
        setResult({
          error: '응답 형식이 올바르지 않습니다.',
          raw_response: JSON.stringify(data)?.slice(0, 1000),
        });
        return;
      }

      setResult(data);

      // ✅ 게이트 통과 시 자동 저장(→ 백엔드가 민팅까지 수행)
      const a = data.result;
      if (a?.gate_pass === true) {
        await savePost({ analysis: a, meta: data.meta });
      } else {
        Alert.alert(
          '게이트 미통과',
          `최종 점수(S_pre)가 설정 임계값(${(a?.gate_used ?? gate).toFixed(2)}) 미만이라 저장하지 않았습니다.`
        );
      }
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

  // ─────────────────────────────────────────────────────────────
  // 렌더
  // ─────────────────────────────────────────────────────────────
  return (
    <KeyboardAvoidingView
      behavior={Platform.select({ ios: 'padding', android: undefined })}
      style={{ flex: 1 }}
    >
      {/* 상단 탭 */}
      <SafeAreaView>
        <View style={[styles.topTabs, { marginTop: 10 }]}>
          <TouchableOpacity
            onPress={() => setTab('analyze')}
            style={[styles.tabBtn, tab === 'analyze' && styles.tabBtnActive]}
          >
            <Text style={[styles.tabTxt, tab === 'analyze' && styles.tabTxtActive]}>분석/등록</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => setTab('community')}
            style={[styles.tabBtn, tab === 'community' && styles.tabBtnActive]}
          >
            <Text style={[styles.tabTxt, tab === 'community' && styles.tabTxtActive]}>커뮤니티</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>

      {tab === 'community' ? (
        // 커뮤니티 화면
        <CommunityApp
          apiBase={backendURL}           // ✅ 상세화면이 같은 백엔드로 본문을 불러오도록 전달
          onBackToAnalyze={() => setTab('analyze')}
        />
      ) : (
        // 분석/등록 화면
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          {/* 상태/디버그 박스 */}
          <View style={styles.debugBox}>
            <Text style={styles.debugTitle}>Backend</Text>
            <Text selectable style={styles.debugText}>
              URL: {backendURL || '(없음)'}
            </Text>
            <Text style={[styles.debugText, { marginTop: 6 }]}>
              📎 PDFs: {filesInfo.count}개 ({filesInfo.sizeLabel})
            </Text>
            <View style={{ marginTop: 8, flexDirection: 'row', gap: 8 }}>
              <Button title="Gate 0.70" onPress={() => setGate(0.70)} />
              <Button title="0.50" onPress={() => setGate(0.50)} />
              <Button title="0.12" onPress={() => setGate(0.12)} />
            </View>
            <Text style={{ color: '#475569', marginTop: 4 }}>
              현재 Gate(정규화): {gate}
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
            style={[styles.input, { height: 120 ]}}
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
              disabled={!canSubmit}
            />
          </View>

          {/* 결과 표시 */}
          {result?.result && (
            <View style={styles.resultBox}>
              <Text style={styles.resultTitle}>📊 분석 결과</Text>

              {/* ❌ 숨김: 최종 점수(S_pre), 커버리지, 토큰 수/매칭, PII, 해시 */}
              {/* ✅ 공개: 진정성, 정확성, 게이트, 통과여부 */}
              <Text>
                진정성(S_sinc): {(result.result.S_sinc * 100).toFixed(1)}점 / 100
                {'  '}(정규화 {(result.result.S_sinc).toFixed(3)})
              </Text>
              <Text>
                정확성(S_fact): {result.result.S_fact == null
                  ? '검증 불가'
                  : `${(result.result.S_fact * 100).toFixed(1)}점 / 100 (정규화 ${result.result.S_fact.toFixed(3)})`}
              </Text>

              {'gate_used' in result.result && (
                <Text>
                  게이트: {(result.result.gate_used * 100).toFixed(1)}점 / 100
                  {'  '}(정규화 {result.result.gate_used.toFixed(3)})
                </Text>
              )}

              <Text>게이트 통과: {result.result.gate_pass ? '✅' : '❌'}</Text>

              {saving && (
                <View style={{ flexDirection:'row', alignItems:'center', gap:8, marginTop:6 }}>
                  <ActivityIndicator />
                  <Text>저장 중…</Text>
                </View>
              )}
              {savedId && (
                <Text style={{ marginTop:6 }}>📌 등록 완료 ID: {savedId}</Text>
              )}

              {/* 자동 민팅 결과 표시(선택) */}
              {mintInfo && (
                <View style={{ marginTop:8, gap:4 }}>
                  <Text>민팅: {String(mintInfo.minted)}</Text>
                  {mintInfo.token_id != null && <Text>token_id: {mintInfo.token_id}</Text>}
                  {mintInfo.tx_hash && <Text numberOfLines={1}>tx: {mintInfo.tx_hash}</Text>}
                  {mintInfo.explorer && (
                    <Button title="Etherscan에서 보기" onPress={() => Linking.openURL(mintInfo.explorer)} />
                  )}
                  {!mintInfo.minted && mintInfo.mint_error && (
                    <Text style={{ color:'#b00020' }}>민팅 오류: {mintInfo.mint_error}</Text>
                  )}
                </View>
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
      )}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  // 상단 탭 스타일
  topTabs: {
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 8,
    backgroundColor: '#f8fafc',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  tabBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#fff',
  },
  tabBtnActive: {
    backgroundColor: '#eef2ff',
    borderColor: '#c7d2fe',
  },
  tabTxt: { color: '#334155', fontWeight: '600' },
  tabTxtActive: { color: '#1d4ed8' },

  // 기존 스타일
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
