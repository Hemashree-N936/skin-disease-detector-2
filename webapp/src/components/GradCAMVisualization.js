import React, { useState } from 'react';
import { Card, Typography, Tabs, Button, Space, Tag } from 'antd';
import { EyeOutlined, DownloadOutlined, ZoomInOutlined, ZoomOutOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const VisualizationCard = styled(Card)`
  .ant-card-body {
    padding: 24px;
  }
`;

const ImageContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
`;

const VisualizationImage = styled.img`
  width: 100%;
  height: auto;
  display: block;
  border-radius: 12px;
`;

const ImageControls = styled.div`
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 16px;
  flex-wrap: wrap;
`;

const InfoContainer = styled.div`
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
`;

const GradCAMVisualization = ({ gradcamData }) => {
  const [activeTab, setActiveTab] = useState('overlay');
  const [imageScale, setImageScale] = useState(1);

  const handleZoomIn = () => {
    setImageScale(prev => Math.min(prev + 0.1, 2));
  };

  const handleZoomOut = () => {
    setImageScale(prev => Math.max(prev - 0.1, 0.5));
  };

  const handleDownload = (imageUrl, filename) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getTabTitle = (key, label) => (
    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {label}
      <Tag color="blue" size="small">
        {key === 'overlay' ? 'Combined' : key === 'heatmap' ? 'Heatmap' : 'Grad-CAM'}
      </Tag>
    </span>
  );

  const getCurrentImage = () => {
    switch (activeTab) {
      case 'heatmap':
        return gradcamData.heatmapImage;
      case 'gradcam':
        return gradcamData.gradcamImage;
      case 'overlay':
      default:
        return gradcamData.overlayImage;
    }
  };

  const getTabDescription = (key) => {
    switch (key) {
      case 'heatmap':
        return 'Shows the activation intensity as a color-coded heatmap. Red areas indicate higher activation.';
      case 'gradcam':
        return 'Class activation map showing which regions contributed most to the prediction.';
      case 'overlay':
        return 'Combination of the original image with the activation heatmap for easy interpretation.';
      default:
        return '';
    }
  };

  return (
    <VisualizationCard>
      <Title level={4} style={{ textAlign: 'center', marginBottom: 24 }}>
        Grad-CAM Visualization
      </Title>

      <InfoContainer>
        <Space direction="vertical" size="small">
          <div>
            <Text strong>Visualization Type:</Text> {getTabTitle(activeTab, 'Current View')}
          </div>
          <div>
            <Text strong>Predicted Disease:</Text> {gradcamData.classLabel.charAt(0).toUpperCase() + gradcamData.classLabel.slice(1)}
          </div>
          <div>
            <Text strong>Confidence:</Text> {(gradcamData.confidence * 100).toFixed(1)}%
          </div>
          <div>
            <Text strong>Description:</Text> {getTabDescription(activeTab)}
          </div>
        </Space>
      </InfoContainer>

      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        centered
        size="large"
      >
        <TabPane tab={getTabTitle('overlay', 'Overlay')} key="overlay">
          <div style={{ textAlign: 'center', marginBottom: 16 }}>
            <Text type="secondary">
              Original image with activation heatmap overlay for easy interpretation
            </Text>
          </div>
        </TabPane>
        
        <TabPane tab={getTabTitle('heatmap', 'Heatmap')} key="heatmap">
          <div style={{ textAlign: 'center', marginBottom: 16 }}>
            <Text type="secondary">
              Pure activation heatmap showing intensity patterns
            </Text>
          </div>
        </TabPane>
        
        <TabPane tab={getTabTitle('gradcam', 'Grad-CAM')} key="gradcam">
          <div style={{ textAlign: 'center', marginBottom: 16 }}>
            <Text type="secondary">
              Class activation map with medical feature highlighting
            </Text>
          </div>
        </TabPane>
      </Tabs>

      <ImageContainer>
        <VisualizationImage 
          src={getCurrentImage()} 
          alt={`Grad-CAM ${activeTab}`}
          style={{ transform: `scale(${imageScale})` }}
        />
      </ImageContainer>

      <ImageControls>
        <Button 
          icon={<ZoomOutOutlined />}
          onClick={handleZoomOut}
          disabled={imageScale <= 0.5}
        >
          Zoom Out
        </Button>
        
        <Button 
          icon={<ZoomInOutlined />}
          onClick={handleZoomIn}
          disabled={imageScale >= 2}
        >
          Zoom In
        </Button>
        
        <Button 
          icon={<DownloadOutlined />}
          onClick={() => handleDownload(getCurrentImage(), `${gradcamData.classLabel}_${activeTab}.jpg`)}
        >
          Download {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
        </Button>
        
        <Button 
          icon={<EyeOutlined />}
          onClick={() => {
            const newWindow = window.open(getCurrentImage(), '_blank');
            newWindow.focus();
          }}
        >
          Open in New Tab
        </Button>
      </ImageControls>

      <div style={{ marginTop: 24, padding: 16, background: '#f8f9fa', borderRadius: 8 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          <strong>Medical Interpretation:</strong> The highlighted regions indicate areas that the AI model 
          focused on when making the prediction. Brighter areas (red/orange) contributed more to the 
          classification decision. This visualization helps clinicians understand the model's reasoning 
          and build trust in the AI-assisted diagnosis.
        </Text>
      </div>
    </VisualizationCard>
  );
};

export default GradCAMVisualization;
