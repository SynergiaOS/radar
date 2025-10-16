# GPW Smart Analyzer - Modern Trading Chart Application

A comprehensive trading analysis application for Warsaw Stock Exchange (GPW) stocks with professional TradingView-style charts, real-time technical indicators, pattern detection, and ML/RL-powered trading signals.

## 🚀 Features

### Core Features
- **TradingView-style Professional Charts** - Interactive candlestick charts with full zoom and pan support
- **Real-time Price Updates** - WebSocket integration for live price monitoring
- **100+ Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, and more
- **Pattern Detection** - Automatic identification of flags, triangles, head & shoulders, double tops/bottoms
- **Advanced Analytics** - ML/RL integration for intelligent trading signals
- **Multi-timeframe Support** - 1D, 1W, 1M, 3M, 6M, 1Y, ALL timeframes
- **Investment Decisions** - Clear KUP/TRZYMAJ/SPRZEDAJ recommendations based on ROE/P/E analysis

### Dashboard Features
- **Real-time Monitoring** - Live price tracking and alert system
- **Comprehensive Analysis** - Process all WIG30/WIG20 stocks simultaneously
- **Signal Distribution** - Visual breakdown of buy/sell/hold recommendations
- **Interactive Filtering** - Filter stocks by decision type, performance metrics
- **Historical Data** - Complete analysis with export capabilities

### Desktop Application
- **Native Desktop App** - Built with Tauri for cross-platform desktop experience
- **System Integration** - Native file system access, system notifications
- **Offline Capabilities** - Local data storage and analysis
- **Performance Optimized** - Fast, responsive native performance

## 🛠 Tech Stack

### Frontend
- **Next.js 14** - Modern React framework with App Router
- **TypeScript** - Type-safe development experience
- **Tailwind CSS + shadcn/ui** - Modern, responsive UI components
- **Lightweight Charts** - TradingView official charting library
- **@ixjb94/indicators** - High-performance technical indicators
- **TanStack Query** - Advanced state management and data fetching
- **react-use-websocket** - Real-time WebSocket integration

### Backend
- **Python Flask** - RESTful API with WebSocket support
- **yfinance** - Real-time market data integration
- **ta library** - Professional technical analysis indicators
- **WebSocket** - Real-time price updates and notifications
- **SQLite/CSV** - Local data storage and analysis results

### Desktop App
- **Tauri** - Cross-platform desktop application framework
- **Rust** - Native performance and system integration
- **Node.js** - Web application runtime

## 📦 Installation

### Prerequisites
- **Node.js 18+** - Frontend development runtime
- **Python 3.8+** - Backend server
- **Rust 1.70+** - Desktop app compilation
- **Git** - Version control

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gpw-smart-analyzer
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Set up Python environment**
   ```bash
   cd ..  # Go to parent directory (radar folder)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install flask flask-socketio yfinance pandas ta
   ```

4. **Install Tauri CLI**
   ```bash
   npm install -g @tauri-apps/cli
   ```

## 🚀 Running the Application

### Development Mode

1. **Start the backend server**
   ```bash
   # In the radar folder (with virtual environment activated)
   python web_dashboard.py
   ```

2. **Start the frontend**
   ```bash
   # In the gpw-smart-analyzer folder
   npm run dev
   ```

3. **Start the desktop app**
   ```bash
   # In the gpw-smart-analyzer folder
   npm run tauri:dev
   ```

### Access Points
- **Web Dashboard**: http://localhost:3000
- **API Server**: http://localhost:5000
- **Desktop App**: Native application window

## 📊 Usage Guide

### Dashboard Navigation
1. **Main Dashboard** - Overview of all stocks with buy/sell/hold signals
2. **Individual Charts** - Click any ticker for detailed analysis with indicators
3. **Real-time Monitoring** - Enable live price updates with WebSocket
4. **Technical Analysis** - Toggle 100+ indicators and patterns
5. **Signal Filtering** - Filter by decision type, performance metrics

### Chart Features
- **Zoom & Pan** - Mouse wheel zoom, click and drag to pan
- **Timeframe Selection** - Choose from multiple timeframes
- **Indicator Overlays** - Add/remove technical indicators dynamically
- **Pattern Detection** - Automatic pattern recognition and alerts
- **Crosshair** - Detailed price information on hover

### Investment Decisions
- **KUP** - Strong buy signal (ROE ≥ 10% AND P/E ≤ 15)
- **TRZYMAJ** - Hold signal (moderate criteria or close to thresholds)
- **SPRZEDAJ** - Sell signal (doesn't meet criteria or unprofitable)

## 🔧 Configuration

### Environment Variables
Create `.env.local` in the gpw-smart-analyzer folder:
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=ws://localhost:5000
NEXT_PUBLIC_DEFAULT_TICKER=XTB.WA
NEXT_PUBLIC_REFRESH_INTERVAL=30000
NEXT_PUBLIC_ENABLE_DEVTOOLS=true
```

### Python Configuration
Modify `config.py` in the radar folder:
```python
# Investment strategy thresholds
ROE_THRESHOLD = 10.0  # Minimum ROE percentage
PE_THRESHOLD = 15.0   # Maximum P/E ratio
ENABLE_DUAL_FILTER = True  # Enable dual filtering
ACTIVE_INDEX = 'WIG30'  # Choose WIG30 or WIG20
```

## 📈 API Endpoints

### Stock Data
- `GET /api/chart/<ticker>` - Chart data with indicators
- `GET /api/indicators/<ticker>` - Technical indicators
- `GET /api/compare` - Compare multiple stocks

### Analysis
- `GET /api/analysis` - Latest analysis results
- `POST /api/run_analysis` - Run new analysis
- `GET /api/all_stocks` - All stocks with decisions
- `GET /api/status` - System status
- `GET /api/config` - Configuration

### WebSocket Events
- `subscribe` - Subscribe to ticker updates
- `unsubscribe` - Unsubscribe from ticker updates
- `price_update` - Real-time price updates

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js App    │    │   Flask API      │    │  Data Sources    │
│                 │    │                 │    │                 │
│ • Dashboard     │◄──►│ • REST API      │◄──►│ • yfinance       │
│ • Charts        │    │ • WebSocket     │    │ • Real-time     │
│ • Indicators    │    │ • Analysis      │    │ • CSV Data      │
│ • Pattern Det.  │    │ • Cache         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Tauri Desktop App                         │
│                                                                 │
│ • Native Performance    • System Integration               │
│ • File System Access     • Cross-platform                      │
│ • Offline Support       • Native Notifications               │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Performance Features

### Frontend Optimization
- **Code Splitting** - Automatic route-based code splitting
- **Lazy Loading** - On-demand component loading
- **Memoization** - Intelligent caching for expensive calculations
- **Virtual Scrolling** - Efficient rendering for large datasets

### Backend Optimization
- **WebSocket Pooling** - Efficient real-time data streaming
- **Data Caching** - Multi-level caching strategy
- **Background Processing** - Non-blocking analysis execution
- **Database Optimization** - Efficient query patterns

### Desktop Performance
- **Native Rendering** - WebKit-based high-performance rendering
- **Memory Management** - Efficient memory usage patterns
- **Startup Optimization** - Fast application launch
- **Background Updates** - Efficient background processing

## 🔒 Security

### Web Security
- **CORS Configuration** - Proper cross-origin resource sharing
- **CSP Headers** - Content Security Policy protection
- **Input Validation** - Server-side input sanitization
- **API Rate Limiting** - Protection against abuse

### Desktop Security
- **Sandboxed Environment** - Isolated application runtime
- **File System Permissions** - Controlled file access
- **Network Security** - Secure HTTP/WebSocket connections
- **Native Integration** - Safe system API usage

## 📱 Cross-Platform Support

### Web Application
- **Chrome/Edge** - Full feature support
- **Firefox** - Full feature support
- **Safari** - Full feature support
- **Mobile Browsers** - Responsive design support

### Desktop Application
- **Windows** - Native Windows application
- **macOS** - Native macOS application
- **Linux** - Native Linux application

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TradingView** - For the excellent Lightweight Charts library
- **yfinance** - For reliable market data access
- **TA Library** - For comprehensive technical indicators
- **Next.js Team** - For the amazing React framework
- **Tauri Team** - For the desktop application framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Join our community discussions

---

**Built with ❤️ for the GPW trading community**