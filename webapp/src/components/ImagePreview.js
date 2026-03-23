import React from 'react';
import { Card, Button, Progress, Typography, Space, Tag } from 'antd';
import { CameraOutlined, CloseOutlined, PlayCircleOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Text, Title } = Typography;

const PreviewContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
`;

const ImageContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 400px;
  aspect-ratio: 1;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  
  @media (max-width: 768px) {
    max-width: 300px;
  }
`;

const PreviewImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const ImageOverlay = styled.div`
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 6px;
  padding: 4px 8px;
  color: white;
  font-size: 12px;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
  
  @media (max-width: 768px) {
    gap: 8px;
  }
`;

const ImageInfo = styled.div`
  text-align: center;
  color: ${props => props.theme.colors.dark};
  
  .file-name {
    font-weight: 500;
    margin-bottom: 4px;
    word-break: break-all;
  }
  
  .file-size {
    font-size: 12px;
    opacity: 0.6;
  }
`;

const ImagePreview = ({ image, onReset, onAnalyze, isLoading, progress }) => {
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileTypeColor = (type) => {
    if (type.includes('jpeg') || type.includes('jpg')) return '#52c41a';
    if (type.includes('png')) return '#1890ff';
    if (type.includes('bmp')) return '#722ed1';
    if (type.includes('tiff')) return '#fa8c16';
    return '#8c8c8c';
  };

  return (
    <PreviewContainer>
      <ImageContainer>
        <PreviewImage src={image.preview} alt={image.name} />
        <ImageOverlay>
          <Tag color={getFileTypeColor(image.file.type)}>
            {image.file.type.split('/')[1].toUpperCase()}
          </Tag>
        </ImageOverlay>
      </ImageContainer>

      <ImageInfo>
        <div className="file-name">{image.name}</div>
        <div className="file-size">{formatFileSize(image.file.size)}</div>
      </ImageInfo>

      {isLoading && (
        <div style={{ width: '100%', maxWidth: '400px' }}>
          <Text strong style={{ marginBottom: '8px', display: 'block' }}>
            Analyzing image...
          </Text>
          <Progress 
            percent={progress} 
            status="active"
            strokeColor={{
              '0%': '#108ee9',
              '100%': '#87d068',
            }}
          />
        </div>
      )}

      <ActionButtons>
        <Button 
          icon={<PlayCircleOutlined />}
          type="primary"
          size="large"
          onClick={() => onAnalyze(false)}
          disabled={isLoading}
          loading={isLoading}
        >
          Quick Analysis
        </Button>
        
        <Button 
          icon={<CameraOutlined />}
          type="primary"
          size="large"
          onClick={() => onAnalyze(true)}
          disabled={isLoading}
          loading={isLoading}
          style={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            border: 'none'
          }}
        >
          Analysis with Grad-CAM
        </Button>
        
        <Button 
          icon={<CloseOutlined />}
          onClick={onReset}
          disabled={isLoading}
          size="large"
        >
          Remove
        </Button>
      </ActionButtons>
    </PreviewContainer>
  );
};

export default ImagePreview;
