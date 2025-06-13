/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  env: {
    NEXT_PUBLIC_APP_NAME: 'RL-LSTM Trading Agent',
    NEXT_PUBLIC_APP_VERSION: '1.0.0'
  }
}

module.exports = nextConfig 