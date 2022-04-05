import './App.css';
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';
import HomePage from "./Components/HomePage";
import Navbar from "./Components/Navbar";
import AboutPage from './Components/AboutPage';
import ModelPage from './Components/ModelPage';
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { createTheme, ThemeProvider, responsiveFontSizes } from '@mui/material/styles';
import { purple } from '@mui/material/colors';

let theme = createTheme({
  palette: {
    type: 'light',
    primary: {
      main: '#1a237e',
    },
    secondary: {
      main: '#e57373',
      light: '#b71c1c',
    },
    background: {
      paper: '#dcedc8',
      default: '#26a69a',
    },
    text: {
      primary: '#311b92',
      secondary: '#1a237e',
    },
    error: {
      main: '#c62828',
    },
  }
});

theme = responsiveFontSizes(theme)

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <div className="App">
          <Navbar />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/model" element={<ModelPage />} />
          </Routes>
        </div>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
