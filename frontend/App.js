// App.js
import React, { useEffect, useState, useMemo } from "react";
import {
  View,
  Text,
  TextInput,
  Button,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from "react-native";

// 고정 배포/예약 도메인이 있으면 여기에(예: https://api.example.com)
const ENV_API_BASE = process.env.EXPO_PUBLIC_API_BASE_URL || "";
// 부트스트랩(LAN) 주소: 예) http://192.168.0.23:8000
const ENV_BOOTSTRAP = process.env.EXPO_PUBLIC_BOOTSTRAP_URL || "";

// JSON fetch 유틸(타임아웃 지원)
async function fetchJSON(url, { method = "GET", body, timeout = 4000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: null };
  } finally {
    clearTimeout(timer);
  }
}

export default function App() {
  const [backendURL, setBackendURL] = useState("");
  const [backendSource, setBackendSource] = useState("-");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // 1) 고정 주소가 있으면 그걸 사용 -> 2) 부트스트랩으로 /ngrok-url 조회
  useEffect(() => {
    async function bootstrap() {
      // 1) 고정 도메인(배포/예약 도메인)
      if (ENV_API_BASE) {
        setBackendURL(ENV_API_BASE.replace(/\/+$/, ""));
        setBackendSource("env");
        return;
      }

      // 2) 부트스트랩(LAN)
      if (!ENV_BOOTSTRAP) {
        setBackendURL("");
        setBackendSource("none");
        Alert.alert("환경설정 필요", "EXPO_PUBLIC_BOOTSTRAP_URL을 .env에 설정하세요.");
        return;
      }
      const bootstrap = ENV_BOOTSTRAP.replace(/\/+$/, "");
      setBackendURL(bootstrap);
      setBackendSource("bootstrap");

      // 2-1) 최신 ngrok/public URL 조회
      const { ok, data } = await fetchJSON(`${bootstrap}/ngrok-url`, { timeout: 3000 });
      if (ok && data?.ngrok_url) {
        setBackendURL(String(data.ngrok_url).replace(/\/+$/, ""));
        setBackendSource(data?.source || "ngrok"); // env | ngrok
      }
    }
    bootstrap();
  }, []);

  const disabled = useMemo(() => !backendURL, [backendURL]);

  async function onPing() {
    if (disabled) return;
    const { ok, status, data } = await fetchJSON(`${backendURL}/hello`, { timeout: 3000 });
    Alert.alert("Ping", ok ? JSON.stringify(data, null, 2) : `실패(status=${status})`);
  }

  async function onRefreshURL() {
    if (!ENV_BOOTSTRAP) return Alert.alert("알림", "BOOTSTRAP 주소가 없습니다.");
    const bootstrap = ENV_BOOTSTRAP.replace(/\/+$/, "");
    const { ok, data } = await fetchJSON(`${bootstrap}/ngrok-url`, { timeout: 3000 });
    if (ok && data?.ngrok_url) {
      setBackendURL(String(data.ngrok_url).replace(/\/+$/, ""));
      setBackendSource(data?.source || "ngrok");
    } else {
      Alert.alert("갱신 실패", "부트스트랩에서 ngrok URL을 가져오지 못했습니다.");
    }
  }

  // (예시) 분석 요청 – 기존 백엔드 /analyze 스펙에 맞춰 사용하세요.
  async function onAnalyze() {
    if (disabled) return;
    setLoading(true);
    setResult(null);
    try {
      const { ok, status, data } = await fetchJSON(`${backendURL}/analyze`, {
        method: "POST",
        body: { text },
        timeout: 10000,
      });
      if (!ok) throw new Error(`HTTP ${status}`);
      setResult(data);
    } catch (e) {
      Alert.alert("오류", "분석 요청에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>EmoTrust Demo</Text>

        {/* 디버그 패널 */}
        <View style={styles.debugBox}>
          <Text style={styles.debugTitle}>Backend</Text>
          <Text style={styles.debugText}>URL: {backendURL || "-"}</Text>
          <Text style={styles.debugText}>Source: {backendSource}</Text>
        </View>

        <View style={styles.row}>
          <Button title="Ping /hello" onPress={onPing} disabled={disabled} />
          <View style={{ width: 12 }} />
          <Button title="URL 새로고침" onPress={onRefreshURL} />
        </View>

        <TextInput
          style={styles.input}
          placeholder="분석할 텍스트를 입력"
          value={text}
          onChangeText={setText}
          multiline
        />

        <Button title="분석요청" onPress={onAnalyze} disabled={disabled || loading || !text.trim()} />

        {loading && <ActivityIndicator style={{ marginTop: 16 }} />}

        {result && (
          <View style={styles.resultBox}>
            <Text style={styles.resultTitle}>결과</Text>
            <Text selectable style={styles.resultText}>{JSON.stringify(result, null, 2)}</Text>
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 16 },
  title: { fontSize: 20, fontWeight: "700" },
  row: { flexDirection: "row", alignItems: "center" },
  input: {
    minHeight: 120,
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 8,
    padding: 12,
  },
  resultBox: {
    marginTop: 16,
    padding: 12,
    backgroundColor: "#f8fafc",
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 8,
  },
  resultTitle: { fontWeight: "700", marginBottom: 6 },
  resultText: { color: "#111827" },
  debugBox: {
    padding: 12,
    backgroundColor: "#f8fafc",
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 8,
  },
  debugTitle: { fontWeight: "700", marginBottom: 6 },
  debugText: { color: "#334155" },
});
