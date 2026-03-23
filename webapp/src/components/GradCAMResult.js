import React from 'react';
import { Card, Typography, Space, Tag } from 'antd';
import { EyeOutlined, FireOutlined, PictureOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const GradCAMResult = ({ result }) => {
  if (!result) return null;

  const prediction = result.prediction || {};
  const gradcam = result.gradcam || {};

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#52c41a'; // green
    if (confidence >= 0.6) return '#faad14'; // orange
    return '#ff4d4f'; // red
  };

  const getClassColor = (className) => {
    const classColors = {
      'benign': '#52c41a', // green
      'malignant': '#ff4d4f', // red
      'acne': '#1890ff', // blue
      'psoriasis': '#722ed1', // purple
      'eczema': '#fa8c16', // orange
    };
    return classColors[className?.toLowerCase()] || '#1890ff';
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <Card 
      title={
        <Space>
          <EyeOutlined />
          <span>Analysis Results</span>
        </Space>
      }
      style={{ marginBottom: '20px' }}
    >
      {/* Prediction Summary */}
      <div style={{ 
        textAlign: 'center', 
        padding: '20px', 
        backgroundColor: '#f5f5f5', 
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <Title level={4} style={{ margin: 0 }}>
          Predicted Class:
        </Title>
        <Title 
          level={2} 
          style={{ 
            margin: '10px 0',
            color: getClassColor(prediction.class || prediction.predicted_class)
          }}
        >
          {prediction.class || prediction.predicted_class || 'Unknown'}
        </Title>
        
        <Space style={{ marginBottom: '10px' }}>
          <Tag color={getConfidenceColor(prediction.confidence || 0)}>
            {getConfidenceLevel(prediction.confidence || 0)} Confidence
          </Tag>
          <Tag color="blue">
            {((prediction.confidence || 0) * 100).toFixed(1)}%
          </Tag>
        </Space>
      </div>

      {/* Grad-CAM Visualization */}
      <div>
        <Title level={4} style={{ textAlign: 'center', marginBottom: '20px' }}>
          <Space>
            <FireOutlined />
            <span>Grad-CAM Visualization</span>
          </Space>
        </Title>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '15px',
          marginBottom: '20px'
        }}>
          {/* Original Image */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              padding: '10px', 
              backgroundColor: '#fafafa', 
              borderRadius: '8px',
              border: '1px solid #d9d9d9'
            }}>
              <PictureOutlined style={{ fontSize: '24px', marginBottom: '8px', color: '#1890ff' }} />
              <div style={{ fontSize: '14px', fontWeight: 'bold' }}>Original Image</div>
              {gradcam.original_image && (
                <img
                  src={`data:image/jpeg;base64,${gradcam.original_image}`}
                  alt="Original"
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: '180px',
                    objectFit: 'contain',
                    borderRadius: '4px',
                    marginTop: '8px'
                  }}
                />
              )}
            </div>
          </div>

          {/* Grad-CAM Overlay */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              padding: '10px', 
              backgroundColor: '#fafafa', 
              borderRadius: '8px',
              border: '1px solid #d9d9d9'
            }}>
              <EyeOutlined style={{ fontSize: '24px', marginBottom: '8px', color: '#52c41a' }} />
              <div style={{ fontSize: '14px', fontWeight: 'bold' }}>Grad-CAM Overlay</div>
              {gradcam.overlay_image && (
                <img
                  src={`data:image/jpeg;base64,${gradcam.overlay_image}`}
                  alt="Grad-CAM Overlay"
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: '180px',
                    objectFit: 'contain',
                    borderRadius: '4px',
                    marginTop: '8px'
                  }}
                />
              )}
            </div>
          </div>

          {/* Heatmap */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              padding: '10px', 
              backgroundColor: '#fafafa', 
              borderRadius: '8px',
              border: '1px solid #d9d9d9'
            }}>
              <FireOutlined style={{ fontSize: '24px', marginBottom: '8px', color: '#ff4d4f' }} />
              <div style={{ fontSize: '14px', fontWeight: 'bold' }}>Heatmap</div>
              {gradcam.heatmap_image && (
                <img
                  src={`data:image/jpeg;base64,${gradcam.heatmap_image}`}
                  alt="Heatmap"
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: '180px',
                    objectFit: 'contain',
                    borderRadius: '4px',
                    marginTop: '8px'
                  }}
                />
              )}
            </div>
          </div>
        </div>

        {/* Technical Details */}
        <div style={{ 
          marginTop: '20px', 
          padding: '15px', 
          backgroundColor: '#fafafa', 
          borderRadius: '8px',
          fontSize: '14px'
        }}>
          <Text strong>Technical Details:</Text>
          <br />
          <Space direction="vertical" size="small" style={{ marginTop: '8px' }}>
            <Text>
              <Text strong>Target Layer:</Text> {gradcam.target_layer || 'Unknown'}
            </Text>
            <Text>
              <Text strong>Class Index:</Text> {prediction.class_idx || 'Unknown'}
            </Text>
            <Text>
              <Text strong>Confidence Score:</Text> {(prediction.confidence || 0).toFixed(4)}
            </Text>
          </Space>
        </div>

        {/* Interpretation Guide */}
        <div style={{ 
          marginTop: '20px', 
          padding: '15px', 
          backgroundColor: '#e6f7ff', 
          borderRadius: '8px',
          border: '1px solid #91d5ff'
        }}>
          <Text strong style={{ color: '#1890ff' }}>
            📊 What the Grad-CAM Shows:
          </Text>
          <br />
          <Text style={{ fontSize: '13px', marginTop: '8px', display: 'block' }}>
            The colored regions (red/yellow) indicate areas that the model focused on 
            when making its prediction. Brighter colors mean higher importance for the classification.
          </Text>
        </div>
      </div>
    </Card>
  );
};

export default GradCAMResult;
