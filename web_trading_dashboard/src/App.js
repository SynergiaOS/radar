import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';

// Components
import Dashboard from './components/Dashboard';
import TradingSignals from './components/TradingSignals';
import RealTimeMonitor from './components/RealTimeMonitor';
import AlertsPanel from './components/AlertsPanel';
import Navigation from './components/Navigation';

// Services
import { TradingApiService } from './services/TradingApiService';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#007acc',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
    info: {
      main: '#2196f3',
    },
  },
  typography: {
    fontFamily: "'Inter', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
    h3: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
          border: '1px solid #333',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #007acc 0%, #005c9f 100%)',
          borderRadius: 8,
          fontWeight: 600,
          textTransform: 'none',
          '&:hover': {
            background: 'linear-gradient(135deg, #005c9f 0%, #004580 100%)',
          },
        },
      },
    },
  },
});

function App() {
  const [signals, setSignals] = useState([]);
  const [realTimeData, setRealTimeData] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    mlModels: false,
    rlAgent: false,
    monitoring: false,
    lastUpdate: null,
  });
  const [currentView, setCurrentView] = useState('dashboard');

  const tradingApi = new TradingApiService();

  useEffect(() => {
    // Initialize system
    initializeSystem();

    // Set up real-time updates
    const interval = setInterval(() => {
      fetchLatestData();
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const initializeSystem = async () => {
    try {
      // Load existing signals
      const savedSignals = await tradingApi.loadSignals();
      setSignals(savedSignals);

      // Load real-time data
      const rtData = await tradingApi.getRealTimeData();
      setRealTimeData(rtData);

      // Load alerts
      const savedAlerts = await tradingApi.getAlerts();
      setAlerts(savedAlerts);

      setSystemStatus(prev => ({
        ...prev,
        mlModels: true,
        rlAgent: true,
        lastUpdate: new Date().toISOString(),
      }));

    } catch (error) {
      console.error('Failed to initialize system:', error);
    }
  };

  const fetchLatestData = async () => {
    try {
      const latestData = await tradingApi.getRealTimeData();
      setRealTimeData(latestData);

      const newAlerts = await tradingApi.getAlerts();
      setAlerts(newAlerts);

      setSystemStatus(prev => ({
        ...prev,
        lastUpdate: new Date().toISOString(),
      }));
    } catch (error) {
      console.error('Failed to fetch latest data:', error);
    }
  };

  const handleRunAnalysis = async () => {
    try {
      const newSignals = await tradingApi.runAnalysis();
      setSignals(newSignals);
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const handleStartMonitoring = () => {
    setSystemStatus(prev => ({ ...prev, monitoring: !prev.monitoring }));
  };

  const handleSendAlerts = () => {
    if (signals.length > 0) {
      alert(`Would send alerts for ${signals.length} signals`);
    }
  };

  const pageVariants = {
    initial: { opacity: 0, x: 100 },
    in: { opacity: 1, x: 0 },
    out: { opacity: 0, x: -100 },
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
          <Navigation
            currentView={currentView}
            onViewChange={setCurrentView}
            systemStatus={systemStatus}
            onRunAnalysis={handleRunAnalysis}
            onStartMonitoring={handleStartMonitoring}
            onSendAlerts={handleSendAlerts}
          />

          <AnimatePresence mode="wait">
            <motion.div
              key={currentView}
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={{ duration: 0.3 }}
            >
              <Routes>
                <Route path="/" element={
                  <Dashboard
                    signals={signals}
                    realTimeData={realTimeData}
                    alerts={alerts}
                    systemStatus={systemStatus}
                  />
                } />
                <Route path="/signals" element={
                  <TradingSignals
                    signals={signals}
                    systemStatus={systemStatus}
                  />
                } />
                <Route path="/monitor" element={
                  <RealTimeMonitor
                    realTimeData={realTimeData}
                    systemStatus={systemStatus}
                  />
                } />
                <Route path="/alerts" element={
                  <AlertsPanel
                    alerts={alerts}
                    signals={signals}
                  />
                } />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;