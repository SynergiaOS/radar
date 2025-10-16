/** @type {import('next').NextConfig} */

const nextConfig = {
  // API rewrites to proxy backend requests and avoid CORS
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://localhost:5000/:path*',
      },
    ];
  },

  // Tauri configuration
  output: 'export',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,

  // Optimization
  swcMinify: true,
  reactStrictMode: true,

  // Environment variables support
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:5000',
    NEXT_PUBLIC_DEFAULT_TICKER: process.env.NEXT_PUBLIC_DEFAULT_TICKER || 'XTB.WA',
    NEXT_PUBLIC_REFRESH_INTERVAL: process.env.NEXT_PUBLIC_REFRESH_INTERVAL || '30000',
    NEXT_PUBLIC_ENABLE_DEVTOOLS: process.env.NEXT_PUBLIC_ENABLE_DEVTOOLS || 'true',
  },

  // Headers for security
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;