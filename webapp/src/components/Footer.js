import React from 'react';
import { Layout, Typography, Space, Row, Col } from 'antd';
import { GithubOutlined, TwitterOutlined, LinkedinOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Footer: AntFooter } = Layout;
const { Text, Link } = Typography;

const StyledFooter = styled(AntFooter)`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  padding: 24px;
  margin-top: auto;
`;

const FooterContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  text-align: center;
`;

const FooterSection = styled.div`
  margin-bottom: 16px;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const FooterTitle = styled(Text)`
  font-weight: 600;
  color: ${props => props.theme.colors.dark};
  margin-bottom: 8px;
  display: block;
`;

const FooterLinks = styled(Space)`
  justify-content: center;
  flex-wrap: wrap;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 8px;
  }
`;

const FooterLink = styled(Link)`
  color: ${props => props.theme.colors.dark};
  opacity: 0.7;
  
  &:hover {
    color: ${props => props.theme.colors.primary};
    opacity: 1;
  }
`;

const SocialLinks = styled(Space)`
  margin-top: 16px;
  
  .social-icon {
    font-size: 18px;
    color: ${props => props.theme.colors.dark};
    opacity: 0.6;
    transition: all 0.3s ease;
    
    &:hover {
      color: ${props => props.theme.colors.primary};
      opacity: 1;
      transform: translateY(-2px);
    }
  }
`;

const FooterText = styled(Text)`
  color: ${props => props.theme.colors.dark};
  opacity: 0.6;
  font-size: 12px;
  
  @media (max-width: 768px) {
    font-size: 11px;
  }
`;

const Footer = () => {
  return (
    <StyledFooter>
      <FooterContent>
        <FooterSection>
          <FooterTitle>SkinAI - Skin Disease Classification</FooterTitle>
          <FooterText>
            AI-powered dermatology assistant for skin disease classification with Grad-CAM visualization
          </FooterText>
        </FooterSection>

        <FooterSection>
          <FooterLinks size="large">
            <FooterLink href="/">Home</FooterLink>
            <FooterLink href="/classifier">Classifier</FooterLink>
            <FooterLink href="/about">About</FooterLink>
            <FooterLink href="/docs">API Docs</FooterLink>
            <FooterLink href="/research">Research</FooterLink>
          </FooterLinks>
        </FooterSection>

        <FooterSection>
          <FooterLinks size="small">
            <FooterLink href="/privacy">Privacy Policy</FooterLink>
            <FooterLink href="/terms">Terms of Service</FooterLink>
            <FooterLink href="/disclaimer">Medical Disclaimer</FooterLink>
            <FooterLink href="/contact">Contact</FooterLink>
          </FooterLinks>
        </FooterSection>

        <FooterSection>
          <SocialLinks>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-icon"
            >
              <GithubOutlined />
            </a>
            <a 
              href="https://twitter.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-icon"
            >
              <TwitterOutlined />
            </a>
            <a 
              href="https://linkedin.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-icon"
            >
              <LinkedinOutlined />
            </a>
          </SocialLinks>
        </FooterSection>

        <FooterSection>
          <FooterText>
            © 2025 SkinAI. All rights reserved. | 
            Built with ❤️ for medical research and education
          </FooterText>
        </FooterSection>
      </FooterContent>
    </StyledFooter>
  );
};

export default Footer;
