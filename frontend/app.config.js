// frontend/app.config.js
require('dotenv').config(); // .env 로드

module.exports = ({ config }) => ({
  ...config,
  extra: {
    ...config.extra,
    EXPO_PUBLIC_API_BASE_URL:
      process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://localhost:8000',
  },
});
