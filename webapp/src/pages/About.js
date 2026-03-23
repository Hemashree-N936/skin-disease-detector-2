import React from 'react';
import { Typography, Card, Row, Col, Timeline, Space, Tag } from 'antd';
import { 
  TeamOutlined, 
  MedicineBoxOutlined, 
  SafetyCertificateOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const { Title, Paragraph, Text } = Typography;

const AboutContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 48px 24px;
  
  @media (max-width: 768px) {
    padding: 32px 16px;
  }
`;

const SectionTitle = styled(Title)`
  text-align: center;
  margin-bottom: 48px;
  color: white;
  
  @media (max-width: 768px) {
    margin-bottom: 32px;
  }
`;

const InfoCard = styled(Card)`
  height: 100%;
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  }
  
  .ant-card-body {
    padding: 32px;
    height: 100%;
    
    @media (max-width: 768px) {
      padding: 24px;
    }
  }
`;

const IconWrapper = styled.div`
  font-size: 48px;
  text-align: center;
  margin-bottom: 24px;
  
  @media (max-width: 768px) {
    font-size: 36px;
  }
`;

const TimelineContainer = styled.div`
  margin-top: 32px;
  
  .ant-timeline-item-content {
    margin-left: 24px;
    
    @media (max-width: 768px) {
      margin-left: 16px;
    }
  }
`;

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
`;

const About = () => {
  const features = [
    {
      icon: '🤖',
      title: 'Advanced AI Models',
      description: 'State-of-the-art deep learning architectures including EfficientNetB4, CNN-ViT hybrids, and GroupDRO for fairness-aware training.',
      details: [
        'EfficientNetB4 for local feature extraction',
        'Vision Transformer for global context',
        'GroupDRO for bias reduction',
        'Grad-CAM for interpretability'
      ]
    },
    {
      icon: '👁️',
      title: 'Interpretable AI',
      description: 'Grad-CAM visualization shows exactly what the AI focuses on, building trust and enabling clinical validation.',
      details: [
        'Class activation maps',
        'Heatmap visualizations',
        'Medical feature highlighting',
        'Confidence scoring'
      ]
    },
    {
      icon: '🏥',
      title: 'Medical Grade',
      description: 'Designed for clinical use with medical accuracy, diverse skin tone representation, and healthcare compliance.',
      details: [
        'Dark skin optimization',
        'Fairness-aware training',
        'Medical validation',
        'Clinical relevance'
      ]
    }
  ];

  const timelineItems = [
    {
      children: (
        <div>
          <Text strong>Data Collection & Preprocessing</Text>
          <Paragraph type="secondary">
            Gathered diverse skin disease datasets with proper medical annotation and skin tone representation.
          </Paragraph>
        </div>
      ),
      color: 'blue'
    },
    {
      children: (
        <div>
          <Text strong>Model Development</Text>
          <Paragraph type="secondary">
            Implemented multiple architectures including CNN, hybrid models, and fairness-aware training approaches.
          </Paragraph>
        </div>
      ),
      color: 'green'
    },
    {
      children: (
        <div>
          <Text strong>Interpretability Integration</Text>
          <Paragraph type="secondary">
            Added Grad-CAM visualization for model transparency and clinical validation support.
          </Paragraph>
        </div>
      ),
      color: 'orange'
    },
    {
      children: (
        <div>
          <Text strong>Fairness & Bias Reduction</Text>
          <Paragraph type="secondary">
            Implemented GroupDRO training to reduce bias and ensure equal performance across skin tones.
          </Paragraph>
        </div>
      ),
      color: 'red'
    },
    {
      children: (
        <div>
          <Text strong>Production Deployment</Text>
          <Paragraph type="secondary">
            Created scalable API backend with monitoring, security, and responsive web interface.
          </Paragraph>
        </div>
      ),
      color: 'purple'
    }
  ];

  const techStack = [
    'PyTorch', 'FastAPI', 'React', 'EfficientNet', 'Vision Transformer', 
    'Grad-CAM', 'GroupDRO', 'Docker', 'Prometheus', 'Grafana'
  ];

  return (
    <AboutContainer>
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <SectionTitle level={2}>About SkinAI</SectionTitle>
        
        <Row gutter={[32, 32]} style={{ marginBottom: 64 }}>
          {features.map((feature, index) => (
            <Col xs={24} md={8} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <InfoCard>
                  <IconWrapper>{feature.icon}</IconWrapper>
                  <Title level={4} style={{ textAlign: 'center', marginBottom: 16 }}>
                    {feature.title}
                  </Title>
                  <Paragraph style={{ textAlign: 'center', marginBottom: 24 }}>
                    {feature.description}
                  </Paragraph>
                  <div>
                    {feature.details.map((detail, idx) => (
                      <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                        <Text style={{ fontSize: 14 }}>{detail}</Text>
                      </div>
                    ))}
                  </div>
                </InfoCard>
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
        <InfoCard style={{ marginBottom: 64 }}>
          <Title level={3} style={{ textAlign: 'center', marginBottom: 32 }}>
            Development Timeline
          </Title>
          <TimelineContainer>
            <Timeline items={timelineItems} />
          </TimelineContainer>
        </InfoCard>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <InfoCard>
          <Title level={3} style={{ textAlign: 'center', marginBottom: 24 }}>
            Technology Stack
          </Title>
          <div style={{ textAlign: 'center' }}>
            <TechStack>
              {techStack.map((tech, index) => (
                <Tag key={index} color="blue" style={{ fontSize: 14 }}>
                  {tech}
                </Tag>
              ))}
            </TechStack>
          </div>
          
          <Row gutter={[32, 32]} style={{ marginTop: 32 }}>
            <Col xs={24} md={12}>
              <div>
                <Title level={4}>
                  <TeamOutlined /> Research Focus
                </Title>
                <Paragraph>
                  Our research focuses on creating fair, interpretable, and accurate AI systems 
                  for dermatological applications, with special emphasis on:
                </Paragraph>
                <ul>
                  <li>Bias reduction and fairness</li>
                  <li>Model interpretability</li>
                  <li>Clinical validation</li>
                  <li>Diverse skin tone representation</li>
                </ul>
              </div>
            </Col>
            <Col xs={24} md={12}>
              <div>
                <Title level={4}>
                  <SafetyCertificateOutlined /> Medical Compliance
                </Title>
                <Paragraph>
                  Built with medical best practices and healthcare standards in mind:
                </Paragraph>
                <ul>
                  <li>HIPAA-compliant data handling</li>
                  <li>Privacy-first design</li>
                  <li>Clinical validation requirements</li>
                  <li>Medical disclaimer and safety</li>
                </ul>
              </div>
            </Col>
          </Row>
        </InfoCard>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <InfoCard style={{ background: 'linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%)' }}>
          <div style={{ textAlign: 'center' }}>
            <Title level={3} style={{ color: '#1890ff', marginBottom: 16 }}>
              Research Contributions
            </Title>
            <Paragraph style={{ fontSize: 16, maxWidth: '800px', margin: '0 auto' }}>
              This project contributes to the field of medical AI by addressing critical challenges 
              in dermatological diagnosis systems, including bias reduction, model interpretability, 
              and fairness across diverse populations. Our work demonstrates how advanced AI techniques 
              can be applied to create more equitable and trustworthy medical decision support systems.
            </Paragraph>
            <Space size="large" style={{ marginTop: 24, justifyContent: 'center' }}>
              <Tag color="blue" icon={<EyeOutlined />}>
                Interpretable AI
              </Tag>
              <Tag color="green" icon={<MedicineBoxOutlined />}>
                Medical Grade
              </Tag>
              <Tag color="orange" icon={<ThunderboltOutlined />}>
                Fairness-Aware
              </Tag>
            </Space>
          </div>
        </InfoCard>
      </motion.div>
    </AboutContainer>
  );
};

export default About;
