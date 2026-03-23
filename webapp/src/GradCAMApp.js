import React, { useState } from 'react';
import { Layout, Typography, Space, Alert, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import ImageUpload from './components/ImageUpload';
import GradCAMResult from './components/GradCAMResult';
import axios from 'axios';

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;

const GradCAMApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (image) => {
    setSelectedImage(image);
    setResult(null);
    setError(null);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await axios.post('http://localhost:8000/predict-gradcam', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      setResult(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      if (err.response) {
        setError(`Server error: ${err.response.data.detail || 'Unknown error'}`);
      } else if (err.request) {
        setError('Network error. Please check if the server is running on http://localhost:8000');
      } else {
        setError('Error analyzing image. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
    setError(null);
    setLoading(false);
  };

  return (
    <Layout style={{ minHeight: '100vh', backgroundColor: '#f0f2f5' }}>
      {/* Header */}
      <Header style={{ 
        background: '#fff', 
        padding: '0 24px', 
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
            🏥 Skin Disease Classification
          </Title>
          <Text style={{ marginLeft: '20px', color: '#666' }}>
            with Grad-CAM Visualization
          </Text>
        </div>
        <Button 
          icon={<ReloadOutlined />} 
          onClick={handleReset}
          disabled={loading}
        >
          Reset
        </Button>
      </Header>

      {/* Content */}
      <Content style={{ padding: '24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          {/* Error Alert */}
          {error && (
            <Alert
              message="Error"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
              style={{ marginBottom: '24px' }}
            />
          )}

          {/* Main Content */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
            {/* Upload Section */}
            <div>
              <div style={{ 
                backgroundColor: '#fff', 
                padding: '24px', 
                borderRadius: '12px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}>
                <Title level={4} style={{ marginBottom: '20px' }}>
                  📤 Upload Image
                </Title>
                <ImageUpload onImageSelect={handleImageSelect} onError={handleError} />
                
                {selectedImage && (
                  <div style={{ marginTop: '24px', textAlign: 'center' }}>
                    <Button
                      type="primary"
                      size="large"
                      onClick={handlePredict}
                      loading={loading}
                      style={{ minWidth: '200px' }}
                    >
                      {loading ? 'Analyzing...' : 'Analyze with Grad-CAM'}
                    </Button>
                  </div>
                )}
              </div>
            </div>

            {/* Results Section */}
            {result && (
              <div>
                <GradCAMResult result={result} />
              </div>
            )}

            {/* Instructions */}
            {!selectedImage && !result && (
              <div style={{ 
                backgroundColor: '#fff', 
                padding: '24px', 
                borderRadius: '12px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}>
                <Title level={4} style={{ marginBottom: '16px' }}>
                  📋 How to Use
                </Title>
                <ol style={{ fontSize: '14px', lineHeight: '1.6', color: '#666' }}>
                  <li>Upload a clear skin disease image using the upload button above</li>
                  <li>Click "Analyze with Grad-CAM" to process the image</li>
                  <li>View the prediction results and Grad-CAM visualization</li>
                  <li>The colored regions show what areas the model focused on</li>
                </ol>
                
                <div style={{ 
                  marginTop: '20px', 
                  padding: '16px', 
                  backgroundColor: '#f6ffed',
                  borderRadius: '8px',
                  border: '1px solid #b7eb8f'
                }}>
                  <Text style={{ color: '#52c41a', fontWeight: 'bold' }}>
                    💡 Tip: Grad-CAM highlights the regions that contributed most to the model's prediction.
                  </Text>
                </div>
              </div>
            )}
          </div>
        </div>
      </Content>

      {/* Footer */}
      <Footer style={{ textAlign: 'center', backgroundColor: '#fff' }}>
        <Text type="secondary">
          Skin Disease Classification with Grad-CAM ©2024 - AI-Powered Medical Analysis
        </Text>
      </Footer>
    </Layout>
  );
};

export default GradCAMApp;
