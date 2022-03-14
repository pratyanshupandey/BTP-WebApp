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

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<HomePage />}/>
          <Route path="/about" element={<AboutPage />}/>
          <Route path="/model" element={<ModelPage />}/>
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
