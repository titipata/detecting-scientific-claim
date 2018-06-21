import PropTypes from 'prop-types';
import React from 'react';
import styled from 'styled-components';

const StyledNavBar = styled.div`
  height: 60px;
  color: white;
  background-color: #1E303C;

  display: flex;
  align-items: center;
`;

const StyledBrandText = styled.div`
  color: white;
  padding: 5px;
  font-weight: bold;
  padding: 15px;
`;

const NavBar = ({ brandname }) => (
  <StyledNavBar>
    <StyledBrandText>
      { brandname }
    </StyledBrandText>
  </StyledNavBar>
);

NavBar.propTypes = {
  brandname: PropTypes.string.isRequired,
};

export default NavBar;
