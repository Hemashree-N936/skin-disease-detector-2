import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, Upload, Button, Spin, Alert, Progress, Typography, Space, Divider, Row, Col } from 'antd';
import { UploadOutlined, CameraOutlined, CheckCircleOutlined, ExclamationCircleOutlined, EyeOutlined } from '@ant-design/icons';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';
import toast from 'react-hot-toast';
import axios from 'axios';

// Import components
import ImagePreview from '../components/ImagePreview';
import PredictionResults from '../components/PredictionResults';
import GradCAMVisualization from '../components/GradCAMVisualization';
import DiseaseInfo from '../components/DiseaseInfo';

const { Title, Text, Paragraph } = Typography;

// Styled components
const ClassifierContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  
  @media (max-width: 768px) {
    padding: 16px;
  }
`;

const UploadCard = styled(Card)`
  .ant-card-body {
    padding: 32px;
    text-align: center;
    
    @media (max-width: 768px) {
      padding: 20px;
    }
  }
`;

const DropzoneContainer = styled.div`
  border: 2px dashed ${props => props.isDragActive ? props.theme.colors.primary : props.theme.colors.info};
  border-radius: 12px;
  padding: 48px 24px;
  background: ${props => props.isDragActive ? 'rgba(24, 144, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)'};
  cursor: pointer;
  transition: all 0.3s ease;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  
  &:hover {
    border-color: ${props => props.theme.colors.primary};
    background: rgba(24, 144, 255, 0.08);
  }
  
  @media (max-width: 768px) {
    padding: 32px 16px;
    min-height: 250px;
  }
`;

const UploadIcon = styled.div`
  font-size: 48px;
  color: ${props => props.theme.colors.info};
  margin-bottom: 16px;
  
  @media (max-width: 768px) {
    font-size: 36px;
  }
`;

const UploadText = styled.div`
  color: ${props => props.theme.colors.dark};
  font-size: 18px;
  font-weight: 500;
  margin-bottom: 8px;
  
  @media (max-width: 768px) {
    font-size: 16px;
  }
`;

const UploadSubtext = styled.div`
  color: ${props => props.theme.colors.dark};
  opacity: 0.6;
  font-size: 14px;
  
  @media (max-width: 768px) {
    font-size: 12px;
  }
`;

const ResultsContainer = styled.div`
  margin-top: 32px;
  
  @media (max-width: 768px) {
    margin-top: 24px;
  }
`;

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Classifier = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [gradcamData, setGradcamData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file (JPG, PNG, etc.)');
        return;
      }

      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB');
        return;
      }

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage({
          file: file,
          preview: e.target.result,
          name: file.name
        });
        setPrediction(null);
        setGradcamData(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  // Analyze image
  const analyzeImage = async (withGradCAM = true) => {
    if (!image) return;

    setIsLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', image.file);

      const endpoint = withGradCAM ? '/predict-with-gradcam' : '/predict';
      
      // Create progress tracking
      const config = {
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      };

      const response = await axios.post(`${API_BASE_URL}${endpoint}`, formData, config);

      if (response.status === 200) {
        const data = response.data;
        
        // Set prediction results
        setPrediction({
          classLabel: data.class_label,
          confidence: data.confidence,
          probabilities: data.probabilities,
          processingTime: data.processing_time,
          timestamp: data.timestamp
        });

        // Set Grad-CAM data if available
        if (withGradCAM && data.gradcam_image) {
          setGradcamData({
            gradcamImage: `data:image/jpeg;base64,${data.gradcam_image}`,
            heatmapImage: `data:image/jpeg;base64,${data.heatmap_image}`,
            overlayImage: `data:image/jpeg;base64,${data.overlay_image}`,
            classLabel: data.class_label,
            confidence: data.confidence,
            classDescription: data.class_description
          });
        }

        toast.success('Image analyzed successfully!');
      } else {
        throw new Error(response.data.detail || 'Analysis failed');
      }
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to analyze image';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  // Reset analysis
  const resetAnalysis = () => {
    setImage(null);
    setPrediction(null);
    setGradcamData(null);
    setError(null);
    setUploadProgress(0);
  };

  return (
    <ClassifierContainer>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Title level={2} style={{ textAlign: 'center', marginBottom: 32, color: 'white' }}>
          Skin Disease Classifier
        </Title>
        
        <Row gutter={[24, 24]}>
          {/* Upload Section */}
          <Col xs={24} lg={12}>
            <UploadCard>
              {!image ? (
                <DropzoneContainer {...getRootProps()} isDragActive={isDragActive}>
                  <input {...getInputProps()} />
                  <UploadIcon>
                    <CameraOutlined />
                  </UploadIcon>
                  <UploadText>
                    {isDragActive ? 'Drop the image here' : 'Upload skin image'}
                  </UploadText>
                  <UploadSubtext>
                    Drag & drop an image here, or click to select
                  </UploadSubtext>
                  <UploadSubtext>
                    Supports: JPG, PNG, BMP, TIFF (Max 10MB)
                  </UploadSubtext>
                </DropzoneContainer>
              ) : (
                <ImagePreview 
                  image={image} 
                  onReset={resetAnalysis}
                  onAnalyze={analyzeImage}
                  isLoading={isLoading}
                  progress={uploadProgress}
                />
              )}
            </UploadCard>
          </Col>

          {/* Results Section */}
          <Col xs={24} lg={12}>
            <AnimatePresence mode="wait">
              {prediction && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <PredictionResults prediction={prediction} />
                  <Divider />
                  <DiseaseInfo disease={prediction.classLabel} />
                </motion.div>
              )}
              
              {error && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Alert
                    message="Analysis Failed"
                    description={error}
                    type="error"
                    showIcon
                    icon={<ExclamationCircleOutlined />}
                  />
                </motion.div>
              )}
              
              {!image && !prediction && !error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <Card>
                    <div style={{ textAlign: 'center', padding: '32px' }}>
                      <EyeOutlined style={{ fontSize: '48px', color: '#1890ff', marginBottom: '16px' }} />
                      <Title level={4}>Upload an image to begin analysis</Title>
                      <Paragraph>
                        Our AI model will analyze the skin image and provide a prediction
                        with confidence scores and Grad-CAM visualization.
                      </Paragraph>
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </Col>
        </Row>

        {/* Grad-CAM Visualization */}
        {gradcamData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <ResultsContainer>
              <GradCAMVisualization gradcamData={gradcamData} />
            </ResultsContainer>
          </motion.div>
        )}
      </motion.div>
    </ClassifierContainer>
  );
};

export default Classifier;
