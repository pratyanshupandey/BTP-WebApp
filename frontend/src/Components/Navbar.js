import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuIcon from '@mui/icons-material/Menu';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import { Link } from 'react-router-dom'

const Navbar = () => {
  const [anchorElNav, setAnchorElNav] = React.useState(null);

  const handleOpenNavMenu = (event) => {
    setAnchorElNav(event.currentTarget);
  };

  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };

  return (
    <AppBar position="sticky">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{ mr: 2, display: { xs: 'none', md: 'flex' } }}
          >
            Intent Detection
          </Typography>


          {/*Handles Click open menu on mobile devices */}
          <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleOpenNavMenu}
              color="inherit"
            >
              <MenuIcon />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorElNav}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'left',
              }}
              open={Boolean(anchorElNav)}
              onClose={handleCloseNavMenu}
              sx={{
                display: { xs: 'block', md: 'none' },
              }}
            >
              <Link to="/Intent_Detection_App/">
                <MenuItem key="home" onClick={handleCloseNavMenu}>
                  <Typography textAlign="center">
                    Home
                  </Typography>
                </MenuItem>
              </Link>
              <Link to="/Intent_Detection_App/model" >
                <MenuItem key="model" onClick={handleCloseNavMenu}>
                  <Typography textAlign="center">
                    Detect Intent
                  </Typography>
                </MenuItem>
              </Link>
              <Link to="/Intent_Detection_App/about" >
                <MenuItem key="about" onClick={handleCloseNavMenu}>
                  <Typography textAlign="center">
                    About Us
                  </Typography>
                </MenuItem>
              </Link>
            </Menu>
          </Box>

          {/*Handles Center name on mobile devices */}
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}
          >
            Intent Detection
          </Typography>

          <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            <Link to="/Intent_Detection_App/">
              <Button
                key="home"
                onClick={handleCloseNavMenu}
                sx={{ my: 2, color: 'white', display: 'block' }}
              >Home
              </Button>
            </Link>
            <Link to="/Intent_Detection_App/model" >
              <Button
                key="model"
                onClick={handleCloseNavMenu}
                sx={{ my: 2, color: 'white', display: 'block' }}
              >Detect Intent
              </Button>
            </Link>
            <Link to="/Intent_Detection_App/about" >
              <Button
                key="about"
                onClick={handleCloseNavMenu}
                sx={{ my: 2, color: 'white', display: 'block' }}
              >About Us
              </Button>
            </Link>
          </Box>

        </Toolbar>
      </Container>
    </AppBar>
  );
};
export default Navbar;
