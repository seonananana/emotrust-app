// components/ScoreGauge.js
import React from 'react';
import { View, Text } from 'react-native';

export default function ScoreGauge({ label, score }) {
  return (
    <View style={{ marginVertical: 8 }}>
      <Text>{label}: {score.toFixed(3)}</Text>
    </View>
  );
}
