import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // TradingView-style dark theme colors
        background: '#131722',
        surface: '#1e1e1e',
        surfaceLight: '#2a2a2a',
        border: '#333333',
        text: {
          primary: '#d1d4dc',
          secondary: '#848e9c',
          muted: '#5d656b',
        },
        // Trading-specific colors
        up: '#26a69a',
        down: '#ef5350',
        neutral: '#757575',
        // Volume colors
        volume: {
          up: '#26a69a80',
          down: '#ef535080',
        },
        // Indicator colors
        indicators: {
          sma20: '#2962FF',
          sma50: '#FF6D00',
          sma200: '#F50057',
          ema12: '#00C853',
          ema26: '#FFAB00',
          rsi: '#FF4081',
          macd: '#7C4DFF',
          bb: '#00B0FF',
          atr: '#FFD600',
          vwap: '#B388FF',
        },
        // Action colors
        actions: {
          buy: '#00E676',
          sell: '#FF5252',
          hold: '#FFC107',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        lg: '12px',
        md: '8px',
        sm: '6px',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-subtle': 'pulseSubtle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
      },
      boxShadow: {
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('tailwindcss-animate'),
  ],
};

export default config;