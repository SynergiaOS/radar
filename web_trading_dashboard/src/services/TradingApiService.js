import axios from 'axios';

class TradingApiService {
  constructor() {
    this.baseURL = 'http://localhost:8000'; // Backend API URL
    this.axios = axios.create({
      baseURL: this.baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Load existing signals from backend
  async loadSignals() {
    try {
      const response = await this.axios.get('/api/signals');
      return response.data.signals || [];
    } catch (error) {
      console.error('Failed to load signals:', error);
      // Return mock data if backend is not available
      return this.getMockSignals();
    }
  }

  // Run analysis
  async runAnalysis() {
    try {
      const response = await this.axios.post('/api/analyze');
      return response.data.signals || [];
    } catch (error) {
      console.error('Analysis failed:', error);
      return this.getMockSignals();
    }
  }

  // Get real-time data
  async getRealTimeData() {
    try {
      const response = await this.axios.get('/api/realtime');
      return response.data;
    } catch (error) {
      console.error('Failed to get real-time data:', error);
      return this.getMockRealTimeData();
    }
  }

  // Get alerts
  async getAlerts() {
    try {
      const response = await this.axios.get('/api/alerts');
      return response.data.alerts || [];
    } catch (error) {
      console.error('Failed to get alerts:', error);
      return this.getMockAlerts();
    }
  }

  // Send alerts
  async sendAlerts(signals) {
    try {
      const response = await this.axios.post('/api/alerts', { signals });
      return response.data;
    } catch (error) {
      console.error('Failed to send alerts:', error);
      return { success: false, message: 'Failed to send alerts' };
    }
  }

  // Get system status
  async getSystemStatus() {
    try {
      const response = await this.axios.get('/api/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get system status:', error);
      return this.getMockSystemStatus();
    }
  }

  // Mock data for development
  getMockSignals() {
    return [
      {
        ticker: 'XTB.WA',
        name: 'XTB S.A.',
        current_price: 67.36,
        final_action: 'STRONG BUY',
        combined_score: 714.3,
        ml_prediction: 'UP',
        ml_confidence: 0.970,
        rl_action: 'BUY',
        rl_confidence: 0.897,
        timestamp: new Date().toISOString(),
      },
      {
        ticker: 'TXT.WA',
        name: 'Text S.A.',
        current_price: 51.00,
        final_action: 'BUY',
        combined_score: 651.6,
        ml_prediction: 'DOWN',
        ml_confidence: 0.650,
        rl_action: 'BUY',
        rl_confidence: 0.823,
        timestamp: new Date().toISOString(),
      },
      {
        ticker: 'PKN.WA',
        name: 'Orlen S.A.',
        current_price: 89.42,
        final_action: 'HOLD',
        combined_score: 45.2,
        ml_prediction: 'NEUTRAL',
        ml_confidence: 0.450,
        rl_action: 'HOLD',
        rl_confidence: 0.620,
        timestamp: new Date().toISOString(),
      },
    ];
  }

  getMockRealTimeData() {
    return {
      'XTB.WA': {
        price: 67.36,
        change: 0.85,
        volume: 15234,
        timestamp: new Date().toISOString(),
      },
      'TXT.WA': {
        price: 51.00,
        change: -0.25,
        volume: 8921,
        timestamp: new Date().toISOString(),
      },
      'PKN.WA': {
        price: 89.42,
        change: 1.23,
        volume: 45678,
        timestamp: new Date().toISOString(),
      },
    };
  }

  getMockAlerts() {
    return [
      {
        id: 1,
        ticker: 'XTB.WA',
        action: 'BUY',
        message: 'Strong buy signal - ML confidence 97%',
        timestamp: new Date().toISOString(),
        type: 'success',
      },
      {
        id: 2,
        ticker: 'TXT.WA',
        action: 'BUY',
        message: 'Buy opportunity - RL agent recommends position',
        timestamp: new Date().toISOString(),
        type: 'info',
      },
    ];
  }

  getMockSystemStatus() {
    return {
      mlModels: true,
      rlAgent: true,
      monitoring: true,
      lastUpdate: new Date().toISOString(),
      uptime: '2h 34m',
      memoryUsage: '45%',
    };
  }
}

export default TradingApiService;