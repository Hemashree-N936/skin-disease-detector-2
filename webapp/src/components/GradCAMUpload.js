import React, { useState } from 'react';
import { Upload, Button, Card, Spin, Alert, Typography, Space } from 'antd';
import { UploadOutlined, ClearOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;

const GradCAMUpload = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = (file) => {
    // Check file type
    const isImage = file.type.startsWith('image/');
    if (!isImage) {
      setError('Please upload an image file (JPG, PNG, etc.)');
      return false;
    }

    // Check file size (max 10MB)
    const isLt10M = file.size / 1024 / 1024 < 10;
    if (!isLt10M) {
      setError('Image must be smaller than 10MB');
      return false;
    }

    setError(null);
    setImage(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);

    return false; // Prevent automatic upload
  };

  const handlePredict = async () => {
    if (!image) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const response = await axios.post('http://localhost:8000/predict-gradcam', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 seconds timeout
      });

      setResult(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      if (err.response) {
        setError(`Server error: ${err.response.data.detail || 'Unknown error'}`);
      } else if (err.request) {
        setError('Network error. Please check if the server is running.');
      } else {
        setError('Error uploading image. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

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
    return classColors[className.toLowerCase()] || '#1890ff';
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '20px' }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: '30px' }}>
        Skin Disease Classification with Grad-CAM
      </Title>

      {/* Upload Section */}
      <Card title="Upload Image" style={{ marginBottom: '20px' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Upload
            beforeUpload={handleImageUpload}
            showUploadList={false}
            accept="image/*"
            style={{ width: '100%' }}
          >
            <Button 
              icon={<UploadOutlined />} 
              size="large"
              style={{ width: '100%' }}
            >
              Choose Image
            </Button>
          </Upload>

          {error && (
            <Alert
              message="Error"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
            />
          )}

          {preview && (
            <div style={{ textAlign: 'center', marginTop: '20px' }}>
              <img
                src={preview}
                alt="Preview"
                style={{
                  maxWidth: '100%',
                  maxHeight: '300px',
                  border: '1px solid #d9d9d9',
                  borderRadius: '8px',
                }}
              />
            </div>
          )}

          {preview && (
            <Space style={{ justifyContent: 'center', width: '100%', marginTop: '20px' }}>
              <Button
                type="primary"
                size="large"
                onClick={handlePredict}
                loading={loading}
                icon={<UploadOutlined />}
              >
                {loading ? 'Analyzing...' : 'Analyze with Grad-CAM'}
              </Button>
              <Button
                size="large"
                onClick={handleClear}
                icon={<ClearOutlined />}
              >
                Clear
              </Button>
            </Space>
          )}
        </Space>
      </Card>

      {/* Results Section */}
      {result && (
        <Card title="Analysis Results" style={{ marginBottom: '20px' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* Prediction Results */}
            <div style={{ textAlign: 'center', padding: '20px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
              <Title level={4} style={{ margin: 0 }}>
                Predicted Class:
              </Title>
              <Title 
                level={2} 
                style={{ 
                  margin: '10px 0',
                  color: getClassColor(result.prediction?.class || result.prediction?.predicted_class)
                }}
              >
                {result.prediction?.class || result.prediction?.predicted_class || 'Unknown'}
              </Title>
              <Title level={5} style={{ margin: 0 }}>
                Confidence:
              </Title>
              <Title 
                level={3} 
                style={{ 
                  margin: '10px 0',
                  color: getConfidenceColor(result.prediction?.confidence || 0)
                }}
              >
                {((result.prediction?.confidence || 0) * 100).toFixed(1)}%
              </Title>
            </div>

            {/* Grad-CAM Visualization */}
            <div style={{ marginTop: '30px' }}>
              <Title level={4} style={{ textAlign: 'center', marginBottom: '20px' }}>
                Grad-CAM Visualization
              </Title>
              
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
                gap: '20px',
                marginBottom: '20px'
              }}>
                {/* Original Image */}
                <div style={{ textAlign: 'center' }}>
                  <Title level={5}>Original Image</Title>
                  <img
                    src={`data:image/jpeg;base64,${result.gradcam?.original_image || ''}`}
                    alt="Original"
                    style={{
                      width: '100%',
                      maxWidth: '250px',
                      height: 'auto',
                      border: '1px solid #d9d9d9',
                      borderRadius: '8px',
                    }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                </div>

                {/* Grad-CAM Overlay */}
                <div style={{ textAlign: 'center' }}>
                  <Title level={5}>Grad-CAM Overlay</Title>
                  <img
                    src={`data:image/jpeg;base64,${result.gradcam?.overlay_image || ''}`}
                    alt="Grad-CAM Overlay"
                    style={{
                      width: '100%',
                      maxWidth: '250px',
                      height: 'auto',
                      border: '1px solid #d9d9d9',
                      borderRadius: '8px',
                    }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                </div>

                {/* Heatmap */}
                <div style={{ textAlign: 'center' }}>
                  <Title level={5}>Heatmap</Title>
                  <img
                    src={`data:image/jpeg;base64,${result.gradcam?.heatmap_image || ''}`}
                    alt="Heatmap"
                    style={{
                      width: '100%',
                      maxWidth: '250px',
                      height: 'auto',
                      border: '1px solid #d9d9d9',
                      borderRadius: '8px',
                    }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
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
                <Text>Target Layer: {result.gradcam?.target_layer || 'Unknown'}</Text>
                <br />
                <Text>Class Index: {result.prediction?.class_idx || 'Unknown'}</Text>
              </div>
            </div>
          </Space>
        </Card>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
        }}>
          <Card style={{ textAlign: 'center', padding: '30px' }}>
            <Spin size="large" />
            <Title level={4} style={{ marginTop: '20px', marginBottom: 0 }}>
              Analyzing image with Grad-CAM...
            </Title>
            <Text type="secondary">
              This may take a few seconds
            </Text>
          </Card>
        </div>
      )}
    </div>
  );
};

export default GradCAMUpload;
