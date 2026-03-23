import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Button, Typography, Space } from 'antd';
import { 
  HomeOutlined, 
  CameraOutlined, 
  InfoCircleOutlined,
  MenuOutlined,
  CloseOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const { Header: AntHeader } = Layout;
const { Title } = Typography;

const StyledHeader = styled(AntHeader)`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 1000;
  
  @media (max-width: 768px) {
    padding: 0 16px;
  }
`;

const HeaderContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1200px;
  margin: 0 auto;
  height: 64px;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  color: inherit;
  
  .logo-icon {
    font-size: 28px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .logo-text {
    font-size: 20px;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    
    @media (max-width: 768px) {
      font-size: 16px;
    }
  }
`;

const Navigation = styled.nav`
  display: flex;
  align-items: center;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const MobileMenuButton = styled(Button)`
  display: none;
  
  @media (max-width: 768px) {
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: transparent;
    box-shadow: none;
    
    &:hover, &:focus {
      background: rgba(0, 0, 0, 0.05);
    }
  }
`;

const MobileMenu = styled.div`
  display: none;
  position: fixed;
  top: 64px;
  left: 0;
  right: 0;
  background: white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 999;
  
  @media (max-width: 768px) {
    display: ${props => props.isOpen ? 'block' : 'none'};
  }
`;

const MobileMenuItem = styled.div`
  padding: 16px 24px;
  border-bottom: 1px solid #f0f0f0;
  
  &:last-child {
    border-bottom: none;
  }
  
  a {
    color: inherit;
    text-decoration: none;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    
    &:hover {
      color: #1890ff;
    }
  }
`;

const ActionButtons = styled(Space)`
  @media (max-width: 768px) {
    display: none;
  }
`;

const Header = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <HomeOutlined />,
      label: 'Home',
      path: '/'
    },
    {
      key: '/classifier',
      icon: <CameraOutlined />,
      label: 'Classifier',
      path: '/classifier'
    },
    {
      key: '/about',
      icon: <InfoCircleOutlined />,
      label: 'About',
      path: '/about'
    }
  ];

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setMobileMenuOpen(false);
  };

  return (
    <>
      <StyledHeader>
        <HeaderContent>
          <Link to="/" style={{ textDecoration: 'none' }}>
            <Logo>
              <span className="logo-icon">🏥</span>
              <span className="logo-text">SkinAI</span>
            </Logo>
          </Link>

          <Navigation>
            <Menu
              mode="horizontal"
              selectedKeys={[location.pathname]}
              style={{ border: 'none', background: 'transparent' }}
            >
              {menuItems.map(item => (
                <Menu.Item key={item.key} icon={item.icon}>
                  <Link to={item.path}>{item.label}</Link>
                </Menu.Item>
              ))}
            </Menu>
          </Navigation>

          <ActionButtons>
            <Button type="primary" icon={<CameraOutlined />}>
              <Link to="/classifier" style={{ textDecoration: 'none', color: 'inherit' }}>
                Start Analysis
              </Link>
            </Button>
          </ActionButtons>

          <MobileMenuButton
            icon={mobileMenuOpen ? <CloseOutlined /> : <MenuOutlined />}
            onClick={toggleMobileMenu}
          />
        </HeaderContent>
      </StyledHeader>

      <MobileMenu isOpen={mobileMenuOpen}>
        {menuItems.map(item => (
          <MobileMenuItem key={item.key}>
            <Link to={item.path} onClick={closeMobileMenu}>
              {item.icon}
              {item.label}
            </Link>
          </MobileMenuItem>
        ))}
        <MobileMenuItem>
          <Link to="/classifier" onClick={closeMobileMenu}>
            <CameraOutlined />
            Start Analysis
          </Link>
        </MobileMenuItem>
      </MobileMenu>
    </>
  );
};

export default Header;
