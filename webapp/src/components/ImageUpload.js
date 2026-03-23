import React, { useState } from 'react';
import { Upload, Button, Alert, Space, Progress } from 'antd';
import { UploadOutlined, ClearOutlined, CameraOutlined } from '@ant-design/icons';

const ImageUpload = ({ onImageSelect, onError }) => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);

  const handleImageUpload = (file) => {
    // Check file type
    const isImage = file.type.startsWith('image/');
    if (!isImage) {
      onError('Please upload an image file (JPG, PNG, etc.)');
      return false;
    }

    // Check file size (max 10MB)
    const isLt10M = file.size / 1024 / 1024 < 10;
    if (!isLt10M) {
      onError('Image must be smaller than 10MB');
      return false;
    }

    // Simulate upload progress
    setUploading(true);
    setUploadProgress(0);
    
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 100);

    // Read file
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      setImage(file);
      onImageSelect(file);
      
      // Complete progress
      setTimeout(() => {
        setUploadProgress(100);
        setUploading(false);
        clearInterval(progressInterval);
      }, 500);
    };
    
    reader.onerror = () => {
      onError('Failed to read image file');
      setUploading(false);
      clearInterval(progressInterval);
    };
    
    reader.readAsDataURL(file);

    return false; // Prevent automatic upload
  };

  const handleClear = () => {
    setImage(null);
    setPreview(null);
    setUploadProgress(0);
    setUploading(false);
    onImageSelect(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div style={{ width: '100%' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Upload Button */}
        <Upload
          beforeUpload={handleImageUpload}
          showUploadList={false}
          accept="image/*"
          style={{ width: '100%' }}
          disabled={uploading}
        >
          <Button 
            icon={<CameraOutlined />} 
            size="large"
            style={{ width: '100%' }}
            loading={uploading}
          >
            {uploading ? 'Uploading...' : 'Choose Skin Image'}
          </Button>
        </Upload>

        {/* Upload Progress */}
        {uploading && (
          <div style={{ marginTop: '10px' }}>
            <Progress 
              percent={uploadProgress} 
              status={uploadProgress === 100 ? 'success' : 'active'}
              size="small"
            />
          </div>
        )}

        {/* Image Preview */}
        {preview && (
          <div style={{ 
            textAlign: 'center', 
            padding: '20px', 
            backgroundColor: '#fafafa',
            borderRadius: '8px',
            border: '1px solid #d9d9d9'
          }}>
            <div style={{ marginBottom: '15px' }}>
              <img
                src={preview}
                alt="Preview"
                style={{
                  maxWidth: '100%',
                  maxHeight: '300px',
                  border: '1px solid #d9d9d9',
                  borderRadius: '8px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}
              />
            </div>
            
            {image && (
              <div style={{ fontSize: '14px', color: '#666' }}>
                <div><strong>File:</strong> {image.name}</div>
                <div><strong>Size:</strong> {formatFileSize(image.size)}</div>
                <div><strong>Type:</strong> {image.type}</div>
              </div>
            )}

            <Button
              type="default"
              size="small"
              onClick={handleClear}
              icon={<ClearOutlined />}
              style={{ marginTop: '15px' }}
            >
              Clear Image
            </Button>
          </div>
        )}

        {/* Instructions */}
        {!preview && !uploading && (
          <div style={{ 
            padding: '20px', 
            backgroundColor: '#f0f2f5',
            borderRadius: '8px',
            textAlign: 'center',
            border: '1px dashed #d9d9d9'
          }}>
            <UploadOutlined style={{ fontSize: '48px', color: '#1890ff', marginBottom: '15px' }} />
            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '10px' }}>
              Upload a skin disease image
            </div>
            <div style={{ fontSize: '14px', color: '#666', marginBottom: '15px' }}>
              Supported formats: JPG, PNG, GIF, WebP<br />
              Maximum file size: 10MB
            </div>
            <div style={{ fontSize: '12px', color: '#999' }}>
              Click the button above or drag and drop an image here
            </div>
          </div>
        )}

        {/* File Requirements */}
        <div style={{ 
          marginTop: '15px', 
          padding: '12px', 
          backgroundColor: '#fff7e6',
          borderRadius: '6px',
          border: '1px solid #ffd591'
        }}>
          <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#d46b08', marginBottom: '5px' }}>
            📸 Image Requirements:
          </div>
          <div style={{ fontSize: '11px', color: '#8c6c1c' }}>
            • Clear, well-lit skin images work best<br />
            • Include the area of concern<br />
            • Avoid blurry or low-quality images<br />
            • Standard photo formats (JPG, PNG) recommended
          </div>
        </div>
      </Space>
    </div>
  );
};

export default ImageUpload;
