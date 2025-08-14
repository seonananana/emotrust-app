// screens/CommunityScreen.js
import React, { useEffect, useState } from 'react';
import { ScrollView, Text, View, Button } from 'react-native';
import { fetchPosts } from '../utils/api';
import PostCard from '../components/PostCard';
import { styles } from '../styles';

export default function CommunityScreen({ switchTab }) {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    (async () => {
      const res = await fetchPosts();
      setPosts(res);
    })();
  }, []);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Community</Text>
      {posts.map((p) => <PostCard key={p.id} p={p} />)}
      <Button title="분석 화면으로" onPress={switchTab} />
    </ScrollView>
  );
}
