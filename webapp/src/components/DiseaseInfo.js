import React from 'react';
import { Card, Typography, Tag, Space, Row, Col } from 'antd';
import { 
  ExclamationCircleOutlined, 
  InfoCircleOutlined, 
  MedicineBoxOutlined,
  ClockCircleOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';

const { Title, Text, Paragraph } = Typography;

const InfoCard = styled(Card)`
  .ant-card-body {
    padding: 20px;
  }
  
  .disease-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }
  
  .disease-icon {
    font-size: 24px;
  }
  
  .severity-tag {
    margin-left: auto;
  }
`;

const FeatureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 16px;
`;

const FeatureItem = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px;
  background: ${props => props.theme.colors.light};
  border-radius: 6px;
  
  .feature-icon {
    color: ${props => props.theme.colors.info};
    margin-top: 2px;
  }
  
  .feature-text {
    font-size: 13px;
    line-height: 1.4;
  }
`;

const TreatmentSection = styled.div`
  margin-top: 16px;
  padding: 12px;
  background: linear-gradient(135deg, #e6f7ff 0%, #f0f9ff 100%);
  border-radius: 8px;
  border-left: 4px solid ${props => props.theme.colors.info};
`;

const DiseaseInfo = ({ disease }) => {
  const diseaseInfo = {
    acne: {
      name: 'Acne',
      icon: '🔴',
      severity: 'Mild to Moderate',
      severityColor: 'orange',
      description: 'Inflammatory skin condition characterized by comedones, papules, pustules, and sometimes cysts.',
      features: [
        'Blackheads and whiteheads (comedones)',
        'Red, inflamed papules and pustules',
        'Oily skin and enlarged pores',
        'Potential scarring',
        'Common on face, chest, and back'
      ],
      causes: [
        'Excess oil production',
        'Hair follicles clogged by oil and dead skin cells',
        'Bacteria (Propionibacterium acnes)',
        'Inflammation',
        'Hormonal changes'
      ],
      treatment: 'Topical retinoids, benzoyl peroxide, salicylic acid, antibiotics, or hormonal therapy depending on severity.',
      disclaimer: 'This is for informational purposes only. Consult a dermatologist for proper diagnosis and treatment.'
    },
    psoriasis: {
      name: 'Psoriasis',
      icon: '🟠',
      severity: 'Mild to Severe',
      severityColor: 'red',
      description: 'Autoimmune condition causing rapid skin cell growth, resulting in thick, scaly, red patches.',
      features: [
        'Well-demarcated red patches',
        'Silvery-white scales',
        'Dry, cracked skin that may bleed',
        'Itching, burning, or soreness',
        'Common on elbows, knees, and scalp'
      ],
      causes: [
        'Autoimmune dysfunction',
        'Genetic predisposition',
        'Environmental triggers',
        'Stress',
        'Infections or injuries'
      ],
      treatment: 'Topical corticosteroids, vitamin D analogues, phototherapy, or systemic medications for severe cases.',
      disclaimer: 'This is for informational purposes only. Consult a dermatologist for proper diagnosis and treatment.'
    },
    eczema: {
      name: 'Eczema',
      icon: '🟣',
      severity: 'Mild to Moderate',
      severityColor: 'purple',
      description: 'Inflammatory condition causing dry, itchy, inflamed skin with various presentations.',
      features: [
        'Dry, sensitive skin',
        'Red, inflamed patches',
        'Intense itching',
        'Weeping or oozing lesions',
        'Thickened, scaly skin (chronic)'
      ],
      causes: [
        'Genetic predisposition',
        'Environmental factors',
        'Immune system dysfunction',
        'Skin barrier defects',
        'Allergies and irritants'
      ],
      treatment: 'Moisturizers, topical corticosteroids, calcineurin inhibitors, and avoiding triggers.',
      disclaimer: 'This is for informational purposes only. Consult a dermatologist for proper diagnosis and treatment.'
    }
  };

  const info = diseaseInfo[disease];
  if (!info) return null;

  return (
    <InfoCard>
      <div className="disease-header">
        <span className="disease-icon">{info.icon}</span>
        <div>
          <Title level={5} style={{ margin: 0 }}>
            {info.name}
          </Title>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {info.description}
          </Text>
        </div>
        <Tag className="severity-tag" color={info.severityColor}>
          {info.severity}
        </Tag>
      </div>

      <Paragraph style={{ marginBottom: 16 }}>
        {info.description}
      </Paragraph>

      <div style={{ marginBottom: 16 }}>
        <Text strong style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <InfoCircleOutlined />
          Key Features
        </Text>
        <FeatureGrid>
          {info.features.map((feature, index) => (
            <FeatureItem key={index}>
              <ExclamationCircleOutlined className="feature-icon" />
              <span className="feature-text">{feature}</span>
            </FeatureItem>
          ))}
        </FeatureGrid>
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text strong style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <MedicineBoxOutlined />
          Common Causes
        </Text>
        <FeatureGrid>
          {info.causes.map((cause, index) => (
            <FeatureItem key={index}>
              <InfoCircleOutlined className="feature-icon" />
              <span className="feature-text">{cause}</span>
            </FeatureItem>
          ))}
        </FeatureGrid>
      </div>

      <TreatmentSection>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
          <MedicineBoxOutlined style={{ color: '#1890ff', marginTop: 2 }} />
          <div>
            <Text strong style={{ color: '#1890ff' }}>Treatment Overview</Text>
            <Paragraph style={{ margin: '4px 0 0 0', fontSize: 13 }}>
              {info.treatment}
            </Paragraph>
          </div>
        </div>
      </TreatmentSection>

      <div style={{ 
        marginTop: 16, 
        padding: 12, 
        background: '#fff7e6', 
        borderRadius: 8,
        borderLeft: '4px solid #faad14'
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
          <ClockCircleOutlined style={{ color: '#faad14', marginTop: 2 }} />
          <div>
            <Text strong style={{ color: '#faad14' }}>Medical Disclaimer</Text>
            <Paragraph style={{ margin: '4px 0 0 0', fontSize: 12 }}>
              {info.disclaimer}
            </Paragraph>
          </div>
        </div>
      </div>
    </InfoCard>
  );
};

export default DiseaseInfo;
