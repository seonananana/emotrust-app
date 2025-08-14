// screens/CommunityScreen.js
import React, { useEffect, useState } from 'react';
import { FlatList } from 'react-native';
import { fetchJSON } from '../utils/api';
import PostCard from '../components/PostCard';

export default function CommunityScreen() {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    fetchJSON('/posts').then(res => {
      if (res.ok) setPosts(res.items);
    });
  }, []);

  return (
    <FlatList
      data={posts}
      keyExtractor={item => item.id.toString()}
      renderItem={({ item }) => <PostCard p={item} />}
    />
  );
}
