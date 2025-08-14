// App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import AnalyzeScreen from './screens/AnalyzeScreen';
import CommunityScreen from './screens/CommunityScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator>
        <Tab.Screen name="분석 요청" component={AnalyzeScreen} />
        <Tab.Screen name="커뮤니티" component={CommunityScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
