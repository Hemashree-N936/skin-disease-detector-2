import React from 'react';
import { Typography, Button, Card, Row, Col, Space, Tag } from 'antd';
import { 
  CameraOutlined, 
  EyeOutlined, 
  MedicineBoxOutlined, 
  SafetyCertificateOutlined,
  ArrowRightOutlined,
  CheckCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const { Title, Paragraph, Text } = Typography;

const HomeContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 48px 24px;
  
  @media (max-width: 768px) {
    padding: 32px 16px;
  }
`;

const HeroSection = styled.div`
  text-align: center;
  margin-bottom: 64px;
  
  @media (max-width: 768px) {
    margin-bottom: 48px;
  }
`;

const HeroTitle = styled(Title)`
  font-size: 48px;
  font-weight: bold;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 24px;
  
  @media (max-width: 768px) {
    font-size: 32px;
  }
`;

const HeroSubtitle = styled(Paragraph)`
  font-size: 20px;
  color: white;
  opacity: 0.9;
  margin-bottom: 32px;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
  
  @media (max-width: 768px) {
    font-size: 16px;
  }
`;

const FeatureCard = styled(Card)`
  height: 100%;
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  
  &:hover {
    transform: translateY(-8px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  }
  
  .ant-card-body {
    padding: 32px;
    height: 100%;
    display: flex;
    flex-direction: column;
    
    @media (max-width: 768px) {
      padding: 24px;
    }
  }
`;

const FeatureIcon = styled.div`
  font-size: 48px;
  margin-bottom: 24px;
  text-align: center;
  
  @media (max-width: 768px) {
    font-size: 36px;
  }
`;

const FeatureTitle = styled(Title)`
  text-align: center;
  margin-bottom: 16px;
  color: ${props => props.theme.colors.dark};
`;

const FeatureDescription = styled(Paragraph)`
  text-align: center;
  color: ${props => props.theme.colors.dark};
  opacity: 0.8;
  flex: 1;
  margin-bottom: 24px;
`;

const StatsSection = styled.div`
  margin: 64px 0;
  padding: 48px 24px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  backdrop-filter: blur(10px);
  
  @media (max-width: 768px) {
    margin: 48px 0;
    padding: 32px 16px;
  }
`;

const StatCard = styled.div`
  text-align: center;
  padding: 24px;
  
  @media (max-width: 768px) {
    padding: 16px;
  }
`;

const StatNumber = styled.div`
  font-size: 48px;
  font-weight: bold;
  color: white;
  margin-bottom: 8px;
  
  @media (max-width: 768px) {
    font-size: 36px;
  }
`;

const StatLabel = styled.div`
  font-size: 16px;
  color: rgba(255, 255, 255, 0.8);
  
  @media (max-width: 768px) {
    font-size: 14px;
  }
`;

const CTASection = styled.div`
  text-align: center;
  margin: 64px 0;
  padding: 48px 24px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  backdrop-filter: blur(10px);
  
  @media (max-width: 768px) {
    margin: 48px 0;
    padding: 32px 16px;
  }
`;

const Home = () => {
  const features = [
    {
      icon: '🏥',
      title: 'AI-Powered Diagnosis',
      description: 'Advanced deep learning models trained on thousands of dermatological images for accurate disease classification.',
      color: '#ff4d4f'
    },
    {
      icon: '👁️',
      title: 'Grad-CAM Visualization',
      description: 'See exactly what the AI focuses on with heatmap visualizations that highlight disease-specific regions.',
      color: '#1890ff'
    },
    {
      icon: '🔬',
      title: 'Medical Research',
      description: 'Built on cutting-edge research with fairness-aware training and bias reduction for diverse skin tones.',
      color: '#52c41a'
    },
    {
      icon: '🛡️',
      title: 'Privacy & Security',
      description: 'Your medical images are processed securely with privacy-first design and HIPAA-compliant practices.',
      color: '#722ed1'
    }
  ];

  const stats = [
    { number: '95%', label: 'Accuracy Rate' },
    { number: '3', label: 'Disease Types' },
    { number: '10K+', label: 'Training Images' },
    { number: '24/7', label: 'Availability' }
  ];

  return (
    <HomeContainer>
      <HeroSection>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <HeroTitle>AI-Powered Skin Disease Classification</HeroTitle>
          <HeroSubtitle>
            Advanced medical AI for accurate dermatological diagnosis with interpretable visualizations
          </HeroSubtitle>
          <Space size="large">
            <Button 
              type="primary" 
              size="large" 
              icon={<CameraOutlined />}
              style={{ 
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)'
              }}
            >
              <Link to="/classifier" style={{ textDecoration: 'none', color: 'inherit' }}>
                Try Classifier
              </Link>
            </Button>
            <Button 
              size="large" 
              icon={<EyeOutlined />}
              style={{ background: 'rgba(255, 255, 255, 0.2)', border: '1px solid rgba(255, 255, 255, 0.3)' }}
            >
              <Link to="/about" style={{ textDecoration: 'none', color: 'white' }}>
                Learn More
              </Link>
            </Button>
          </Space>
        </motion.div>
      </HeroSection>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Row gutter={[32, 32]}>
          {features.map((feature, index) => (
            <Col xs={24} sm={12} lg={6} key={index}>
              <motion.div
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.2 }}
              >
                <FeatureCard>
                  <FeatureIcon>{feature.icon}</FeatureIcon>
                  <FeatureTitle level={4}>{feature.title}</FeatureTitle>
                  <FeatureDescription>{feature.description}</FeatureDescription>
                  <div style={{ textAlign: 'center' }}>
                    <Tag color={feature.color} style={{ fontSize: 12 }}>
                      <CheckCircleOutlined /> Available
                    </Tag>
                  </div>
                </FeatureCard>
              </motion.div>
            </Col>
          ))}
        </Row>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <StatsSection>
          <Row gutter={[32, 32]}>
            {stats.map((stat, index) => (
              <Col xs={12} sm={6} key={index}>
                <StatCard>
                  <StatNumber>{stat.number}</StatNumber>
                  <StatLabel>{stat.label}</StatLabel>
                </StatCard>
              </Col>
            ))}
          </Row>
        </StatsSection>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <CTASection>
          <Title level={3} style={{ color: 'white', marginBottom: 16 }}>
            Ready to Experience AI-Powered Dermatology?
          </Title>
          <Paragraph style={{ color: 'rgba(255, 255, 255, 0.8)', marginBottom: 32, fontSize: 16 }}>
            Upload a skin image and get instant AI-powered diagnosis with interpretable visualizations
          </Paragraph>
          <Button 
            type="primary" 
            size="large" 
            icon={<ThunderboltOutlined />}
            style={{ 
              background: 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)',
              border: 'none',
              boxShadow: '0 4px 12px rgba(82, 196, 26, 0.4)'
            }}
          >
            <Link to="/classifier" style={{ textDecoration: 'none', color: 'inherit' }}>
              Start Analysis Now <ArrowRightOutlined />
            </Link>
          </Button>
        </CTASection>
      </motion.div>
    </HomeContainer>
  );
};

export default Home;
