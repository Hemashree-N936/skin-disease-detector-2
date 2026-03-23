import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigProvider, theme } from 'antd';
import { Toaster } from 'react-hot-toast';
import styled, { ThemeProvider } from 'styled-components';
import { motion } from 'framer-motion';

// Import components
import Header from './components/Header';
import Home from './pages/Home';
import Classifier from './pages/Classifier';
import About from './pages/About';
import Footer from './components/Footer';

// Styled components
const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
`;

const MainContent = styled.main`
  flex: 1;
  padding: 0;
  margin: 0;
`;

// Custom theme for medical application
const medicalTheme = {
  colors: {
    primary: '#1890ff',
    secondary: '#52c41a',
    danger: '#ff4d4f',
    warning: '#faad14',
    info: '#13c2c2',
    light: '#f5f5f5',
    dark: '#262626',
    medical: '#722ed1',
    success: '#52c41a'
  },
  fonts: {
    primary: '"Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
    medical: '"Inter", "Segoe UI", sans-serif'
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    xxl: '48px'
  },
  borderRadius: '8px',
  shadows: {
    small: '0 2px 4px rgba(0, 0, 0, 0.1)',
    medium: '0 4px 8px rgba(0, 0, 0, 0.15)',
    large: '0 8px 16px rgba(0, 0, 0, 0.2)'
  }
};

// Ant Design theme configuration
const antdTheme = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: medicalTheme.colors.primary,
    colorSuccess: medicalTheme.colors.success,
    colorWarning: medicalTheme.colors.warning,
    colorError: medicalTheme.colors.danger,
    colorInfo: medicalTheme.colors.info,
    fontFamily: medicalTheme.fonts.primary,
    borderRadius: 8,
    fontSize: 14,
  },
  components: {
    Button: {
      borderRadius: 8,
      fontWeight: 500,
    },
    Card: {
      borderRadius: 12,
      boxShadow: medicalTheme.shadows.medium,
    },
    Upload: {
      borderRadius: 8,
    },
    Progress: {
      borderRadius: 8,
    },
  },
};

function App() {
  const [isLoading, setIsLoading] = useState(false);

  return (
    <ThemeProvider theme={medicalTheme}>
      <ConfigProvider theme={antdTheme}>
        <AppContainer>
          <Router>
            <Header />
            <MainContent>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/classifier" element={<Classifier />} />
                  <Route path="/about" element={<About />} />
                </Routes>
              </motion.div>
            </MainContent>
            <Footer />
          </Router>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: medicalTheme.colors.dark,
                color: medicalTheme.colors.light,
                borderRadius: medicalTheme.borderRadius,
                boxShadow: medicalTheme.shadows.large,
              },
              success: {
                iconTheme: {
                  primary: medicalTheme.colors.success,
                  secondary: medicalTheme.colors.light,
                },
              },
              error: {
                iconTheme: {
                  primary: medicalTheme.colors.danger,
                  secondary: medicalTheme.colors.light,
                },
              },
            }}
          />
        </AppContainer>
      </ConfigProvider>
    </ThemeProvider>
  );
}

export default App;
