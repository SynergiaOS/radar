import React, { useState } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Avatar,
  Chip,
  LinearProgress,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  ShowChart,
  NotificationsActive,
  Speed,
  Assessment,
  Refresh,
  MoreVert,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import ApexChart from 'react-apexcharts';
import { format } from 'date-fns';

const Dashboard = ({ signals, realTimeData, alerts, systemStatus }) => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('1D');

  // Calculate metrics
  const buySignals = signals.filter(s => s.final_action?.includes('BUY')).length;
  const sellSignals = signals.filter(s => s.final_action?.includes('SELL')).length;
  const holdSignals = signals.filter(s => s.final_action === 'HOLD').length;

  const avgScore = signals.length > 0
    ? signals.reduce((acc, s) => acc + s.combined_score, 0) / signals.length
    : 0;

  const topSignals = signals.slice(0, 5);

  // Chart options
  const priceChartOptions = {
    chart: {
      type: 'candlestick',
      background: '#1a1a1a',
      foreColor: '#fff',
      toolbar: {
        show: false,
      },
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#4caf50',
          downward: '#f44336',
        },
      },
    },
    xaxis: {
      type: 'datetime',
      labels: {
        style: {
          colors: '#fff',
        },
      },
    },
    yaxis: {
      tooltip: {
        style: {
          colors: '#fff',
        },
      },
      labels: {
        style: {
          colors: '#fff',
        },
      },
    },
    theme: {
      mode: 'dark',
    },
  };

  // Generate sample candlestick data
  const generateCandlestickData = () => {
    const data = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
      const date = new Date(now - i * 60 * 60 * 1000);
      const basePrice = 67.36 + Math.random() * 10 - 5;

      data.push({
        x: date.getTime(),
        y: [
          basePrice - Math.random() * 2,
          basePrice + Math.random() * 3,
          basePrice - Math.random() * 1,
          basePrice + Math.random() * 2,
        ],
      });
    }
    return data;
  };

  const pieChartOptions = {
    chart: {
      type: 'pie',
      background: 'transparent',
      foreColor: '#fff',
    },
    plotOptions: {
      pie: {
        expandOnClick: false,
        donut: {
          size: '70%',
          labels: {
            show: true,
            total: {
              show: true,
              label: 'Signals',
              color: '#fff',
              fontSize: 16,
              fontWeight: 600,
            },
          },
        },
      },
    },
    labels: ['BUY', 'SELL', 'HOLD'],
    colors: ['#4caf50', '#f44336', '#ff9800'],
    theme: {
      mode: 'dark',
    },
  };

  const lineChartOptions = {
    chart: {
      type: 'line',
      background: '#1a1a1a',
      foreColor: '#fff',
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width: 2,
      curve: 'smooth',
    },
    fill: {
      type: 'gradient',
      gradient: {
        shade: 'dark',
        opacityFrom: 0.6,
        opacityTo: 0.1,
      },
    },
    xaxis: {
      type: 'datetime',
      labels: {
        style: {
          colors: '#fff',
        },
      },
    },
    yaxis: {
      labels: {
        style: {
          colors: '#fff',
        },
      },
    },
    theme: {
      mode: 'dark',
    },
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 700, color: '#fff' }}>
          🤖 ML/RL Trading Dashboard
        </Typography>
      </motion.div>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar sx={{ bgcolor: '#4caf50', mr: 2 }}>
                    <TrendingUp />
                  </Avatar>
                  <Typography variant="h6" color="#fff">
                    Buy Signals
                  </Typography>
                </Box>
                <Typography variant="h3" color="#4caf50" sx={{ fontWeight: 700 }}>
                  {buySignals}
                </Typography>
                <Typography variant="caption" color="#b0b0b0">
                  {buySignals > 0 ? `${Math.round((buySignals / signals.length) * 100)}% of total` : 'No signals'}
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar sx={{ bgcolor: '#f44336', mr: 2 }}>
                    <TrendingDown />
                  </Avatar>
                  <Typography variant="h6" color="#fff">
                    Sell Signals
                  </Typography>
                </Box>
                <Typography variant="h3" color="#f44336" sx={{ fontWeight: 700 }}>
                  {sellSignals}
                </Typography>
                <Typography variant="caption" color="#b0b0b0">
                  {sellSignals > 0 ? `${Math.round((sellSignals / signals.length) * 100)}% of total` : 'No signals'}
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar sx={{ bgcolor: '#2196f3', mr: 2 }}>
                    <Speed />
                  </Avatar>
                  <Typography variant="h6" color="#fff">
                    Avg Score
                  </Typography>
                </Box>
                <Typography variant="h3" color="#2196f3" sx={{ fontWeight: 700 }}>
                  {avgScore.toFixed(1)}
                </Typography>
                <Typography variant="caption" color="#b0b0b0">
                  Combined ML+RL Score
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar sx={{ bgcolor: '#ff9800', mr: 2 }}>
                    <Assessment />
                  </Avatar>
                  <Typography variant="h6" color="#fff">
                    System Status
                  </Typography>
                </Box>
                <Typography variant="h3" color="#4caf50" sx={{ fontWeight: 700 }}>
                  {systemStatus.monitoring ? 'Active' : 'Ready'}
                </Typography>
                <Typography variant="caption" color="#b0b0b0">
                  Last Update: {format(new Date(systemStatus.lastUpdate || new Date()), 'HH:mm:ss')}
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
                height: 400,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" color="#fff">
                    📊 Price Chart - XTB.WA
                  </Typography>
                  <IconButton color="#fff">
                    <Refresh />
                  </IconButton>
                </Box>
                <ApexChart
                  options={priceChartOptions}
                  series={[{ data: generateCandlestickData() }]}
                  type="candlestick"
                  height={300}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={4}>
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
                height: 400,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h6" color="#fff" sx={{ mb: 2 }}>
                  📊 Signal Distribution
                </Typography>
                <ApexChart
                  options={pieChartOptions}
                  series={[buySignals, sellSignals, holdSignals]}
                  type="pie"
                  height={280}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h6" color="#fff" sx={{ mb: 2 }}>
                  📈 Signal Strength Over Time
                </Typography>
                <ApexChart
                  options={lineChartOptions}
                  series={[{
                    name: 'ML Score',
                    data: Array.from({ length: 20 }, (_, i) => ({
                      x: new Date(Date.now() - i * 60 * 60 * 1000).getTime(),
                      y: 60 + Math.random() * 30 + Math.sin(i / 3) * 10,
                    })),
                  }]}
                  type="line"
                  height={200}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Top Signals */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1 }}
          >
            <Card
              sx={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333',
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ p: 2 }}>
                <Typography variant="h6" color="#fff" sx={{ mb: 2 }}>
                  🎯 Top Trading Signals
                </Typography>
                <Grid container spacing={2}>
                  {topSignals.map((signal, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        transition={{ type: 'spring', stiffness: 300 }}
                      >
                        <Paper
                          sx={{
                            p: 2,
                            background: 'rgba(255, 255, 255, 0.05)',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: 1,
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="h6" color="#fff">
                              {signal.ticker}
                            </Typography>
                            <Chip
                              size="small"
                              label={signal.final_action}
                              color={
                                signal.final_action?.includes('BUY') ? 'success' :
                                signal.final_action?.includes('SELL') ? 'error' : 'warning'
                              }
                            />
                          </Box>
                          <Typography variant="body2" color="#b0b0b0" sx={{ mb: 1 }}>
                            {signal.name}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="caption" color="#4caf50">
                              Price: {signal.current_price?.toFixed(2)} PLN
                            </Typography>
                            <Typography variant="caption" color="#2196f3">
                              Score: {signal.combined_score?.toFixed(1)}
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                            <Typography variant="caption" color="#ff9800">
                              ML: {signal.ml_confidence?.toFixed(3)}
                            </Typography>
                            <Typography variant="caption" color="#f44336">
                              RL: {signal.rl_confidence?.toFixed(3)}
                            </Typography>
                          </Box>
                        </Paper>
                      </motion.div>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;