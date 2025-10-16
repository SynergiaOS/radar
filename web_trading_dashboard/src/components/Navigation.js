import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Chip,
  Menu,
  MenuItem,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Dashboard,
  TrendingUp,
  Timeline,
  NotificationsActive,
  Settings,
  Refresh,
  Menu as MenuIcon,
  LightMode,
  DarkMode,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const Navigation = ({
  currentView,
  onViewChange,
  systemStatus,
  onRunAnalysis,
  onStartMonitoring,
  onSendAlerts,
}) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [profileAnchorEl, setProfileAnchorEl] = React.useState(null);

  const menuItems = [
    { text: 'Dashboard', icon: <Dashboard />, path: '/' },
    { text: 'Trading Signals', icon: <TrendingUp />, path: '/signals' },
    { text: 'Real-Time Monitor', icon: <Timeline />, path: '/monitor' },
    { text: 'Alerts', icon: <NotificationsActive />, path: '/alerts' },
  ];

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMenuClick = (path) => {
    onViewChange(path);
    navigate(path);
    setMobileOpen(false);
  };

  const handleProfileMenuOpen = (event) => {
    setProfileAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setProfileAnchorEl(null);
  };

  const getStatusColor = () => {
    if (systemStatus.monitoring) return '#4caf50';
    if (systemStatus.mlModels && systemStatus.rlAgent) return '#2196f3';
    return '#ff9800';
  };

  const getStatusText = () => {
    if (systemStatus.monitoring) return '🟢 Monitoring Active';
    if (systemStatus.mlModels && systemStatus.rlAgent) return '🔵 Systems Ready';
    return '🟡 Initializing';
  };

  const drawer = (
    <Box>
      <Toolbar sx={{ backgroundColor: theme.palette.background.paper, borderRight: 1, borderColor: 'divider' }}>
        <List>
          {menuItems.map((item, index) => (
            <ListItem
              key={item.text}
              button
              selected={location.pathname === item.path}
              onClick={() => handleMenuClick(item.path)}
              sx={{
                mx: 1,
                my: 0.5,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.15),
                  },
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: location.pathname === item.path ? 'primary.main' : 'text.secondary',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                sx={{
                  color: location.pathname === item.path ? 'primary.main' : 'text.secondary',
                  fontWeight: location.pathname === item.path ? 600 : 400,
                }}
              />
            </ListItem>
          ))}
        </List>
      </Toolbar>
    </Box>
  );

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          background: `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${alpha(theme.palette.background.default, 0.8)} 100%)`,
          backdropFilter: 'blur(10px)',
          borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { xs: 'flex', md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>

          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{
              display: { xs: 'none', md: 'flex' },
              flexGrow: 1,
              fontWeight: 600,
              background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              color: 'transparent',
              ml: 2,
              textShadow: '0 2px 4px rgba(0,0,0,0.2)',
            }}
          >
            🤖 Advanced Trading System
          </Typography>

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexGrow: 1, justifyContent: 'flex-end' }}>
            {/* System Status */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <Chip
                avatar={<Avatar sx={{ bgcolor: getStatusColor(), width: 24, height: 24 }}>🔴</Avatar>}
                label={getStatusText()}
                size="small"
                sx={{
                  fontWeight: 600,
                  '& .MuiChip-label': {
                    color: 'white',
                  },
                }}
              />
            </motion.div>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1, duration: 0.3 }}
            >
              <IconButton
                color="inherit"
                onClick={onRunAnalysis}
                sx={{ mx: 1 }}
                title="Run Analysis"
              >
                <Refresh />
              </IconButton>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.3 }}
            >
              <IconButton
                color="inherit"
                onClick={onStartMonitoring}
                sx={{ mx: 1 }}
                title={systemStatus.monitoring ? 'Stop Monitoring' : 'Start Monitoring'}
              >
                <Timeline />
              </IconButton>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.3 }}
            >
              <IconButton
                color="inherit"
                onClick={onSendAlerts}
                sx={{ mx: 1 }}
                title="Send Alerts"
              >
                <NotificationsActive />
              </IconButton>
            </motion.div>

            {/* Profile Menu */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.3 }}
            >
              <IconButton
                color="inherit"
                onClick={handleProfileMenuOpen}
                sx={{ ml: 1 }}
              >
                <Settings />
              </IconButton>
            </motion.div>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Box
        component="nav"
        sx={{ width: 240, flexShrink: { md: 0 }, display: { xs: 'block', md: 'none' } }}
      >
        <Drawer
          variant="temporary"
          anchor="left"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            '& .MuiDrawer-paper': {
              backgroundColor: theme.palette.background.paper,
              borderRight: '1px solid rgba(255, 255, 255, 0.12)',
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Profile Menu */}
      <Menu
        anchorEl={profileAnchorEl}
        open={Boolean(profileAnchorEl)}
        onClose={handleProfileMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem onClick={handleProfileMenuClose}>Profile</MenuItem>
        <MenuItem onClick={handleProfileMenuClose}>Settings</MenuItem>
        <MenuItem onClick={handleProfileMenuClose}>About</MenuItem>
      </Menu>
    </Box>
  );
};

export default Navigation;