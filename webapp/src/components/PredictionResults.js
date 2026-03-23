import React from 'react';
import { Card, Progress, Typography, Tag, Space, Row, Col } from 'antd';
import { CheckCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';

const { Text, Title } = Typography;

const ResultsCard = styled(Card)`
  .ant-card-body {
    padding: 24px;
  }
`;

const ConfidenceContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
`;

const ConfidenceCircle = styled.div`
  width: 120px;
  height: 120px;
  
  @media (max-width: 768px) {
    width: 100px;
    height: 100px;
  }
`;

const ConfidenceText = styled.div`
  text-align: center;
  margin-left: 24px;
  
  @media (max-width: 768px) {
    margin-left: 16px;
  }
`;

const ConfidenceLabel = styled.div`
  font-size: 14px;
  color: ${props => props.theme.colors.dark};
  opacity: 0.6;
  margin-bottom: 4px;
`;

const ConfidenceValue = styled.div`
  font-size: 32px;
  font-weight: bold;
  color: ${props => props.theme.colors.primary};
  
  @media (max-width: 768px) {
    font-size: 24px;
  }
`;

const ProbabilityContainer = styled.div`
  margin-top: 16px;
`;

const ProbabilityItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const ProbabilityLabel = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
`;

const ProbabilityBar = styled.div`
  flex: 2;
  height: 8px;
  background: ${props => props.theme.colors.light};
  border-radius: 4px;
  overflow: hidden;
  margin: 0 12px;
`;

const ProbabilityFill = styled.div`
  height: 100%;
  background: ${props => props.color};
  border-radius: 4px;
  transition: width 0.5s ease;
`;

const ProbabilityValue = styled.div`
  min-width: 50px;
  text-align: right;
  font-weight: 500;
  color: ${props => props.theme.colors.dark};
`;

const ProcessingInfo = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid ${props => props.theme.colors.light};
  font-size: 12px;
  color: ${props => props.theme.colors.dark};
  opacity: 0.6;
`;

const PredictionResults = ({ prediction }) => {
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#52c41a';
    if (confidence >= 0.6) return '#faad14';
    return '#ff4d4f';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  const getClassColor = (className) => {
    const colors = {
      'acne': '#ff4d4f',
      'psoriasis': '#fa8c16',
      'eczema': '#722ed1'
    };
    return colors[className] || '#1890ff';
  };

  const getClassIcon = (className) => {
    const icons = {
      'acne': '🔴',
      'psoriasis': '🟠',
      'eczema': '🟣'
    };
    return icons[className] || '🔵';
  };

  const formatProcessingTime = (time) => {
    return `${(time * 1000).toFixed(0)}ms`;
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <ResultsCard>
      <Title level={4} style={{ textAlign: 'center', marginBottom: 24 }}>
        Analysis Results
      </Title>

      {/* Confidence Circle */}
      <ConfidenceContainer>
        <ConfidenceCircle>
          <CircularProgressbar
            value={prediction.confidence * 100}
            text={`${(prediction.confidence * 100).toFixed(1)}%`}
            styles={buildStyles({
              textSize: '16px',
              pathColor: getConfidenceColor(prediction.confidence),
              textColor: '#262626',
              trailColor: '#f0f0f0',
              strokeLinecap: 'round',
            })}
          />
        </ConfidenceCircle>
        
        <ConfidenceText>
          <ConfidenceLabel>Prediction Confidence</ConfidenceLabel>
          <ConfidenceValue>{prediction.classLabel.toUpperCase()}</ConfidenceValue>
          <Tag 
            color={getConfidenceColor(prediction.confidence)}
            style={{ marginTop: 8 }}
          >
            {getConfidenceLabel(prediction.confidence)} Confidence
          </Tag>
        </ConfidenceText>
      </ConfidenceContainer>

      {/* Class Probabilities */}
      <ProbabilityContainer>
        <Title level={5} style={{ marginBottom: 16 }}>
          Class Probabilities
        </Title>
        
        {Object.entries(prediction.probabilities).map(([className, probability]) => (
          <ProbabilityItem key={className}>
            <ProbabilityLabel>
              <span style={{ fontSize: '16px' }}>
                {getClassIcon(className)}
              </span>
              <span style={{ 
                fontWeight: className === prediction.classLabel ? 'bold' : 'normal',
                color: className === prediction.classLabel ? getClassColor(className) : 'inherit'
              }}>
                {className.charAt(0).toUpperCase() + className.slice(1)}
              </span>
              {className === prediction.classLabel && (
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              )}
            </ProbabilityLabel>
            
            <ProbabilityBar>
              <ProbabilityFill 
                color={getClassColor(className)}
                style={{ width: `${probability * 100}%` }}
              />
            </ProbabilityBar>
            
            <ProbabilityValue>
              {(probability * 100).toFixed(1)}%
            </ProbabilityValue>
          </ProbabilityItem>
        ))}
      </ProbabilityContainer>

      {/* Processing Information */}
      <ProcessingInfo>
        <Space>
          <ClockCircleOutlined />
          <span>Processing time: {formatProcessingTime(prediction.processingTime)}</span>
        </Space>
        <span>{formatTimestamp(prediction.timestamp)}</span>
      </ProcessingInfo>
    </ResultsCard>
  );
};

export default PredictionResults;
