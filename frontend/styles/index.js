// styles/index.js
import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  container: { padding: 20, paddingTop: 60 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 8 },
  label: { fontSize: 14, marginTop: 12 },
  input: { borderWidth: 1, borderColor: '#ccc', padding: 8, borderRadius: 6 },
  resultBox: { marginTop: 16, padding: 12, backgroundColor: '#ecfdf5', borderRadius: 8 },
  card: { padding: 12, backgroundColor: '#f9fafb', borderRadius: 10, marginTop: 12 },
  meta: { fontSize: 13, color: '#64748b' },
  minted: { marginTop: 4, color: 'green', fontWeight: 'bold' },
  notMinted: { marginTop: 4, color: 'gray', fontStyle: 'italic' },
  preview: { fontSize: 13, marginTop: 4 },
});
