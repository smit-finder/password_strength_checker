import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';  // Your CSS file for styling
import PasswordChecker from './App';  // Main app component
import 'bootstrap/dist/css/bootstrap.min.css';


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <PasswordChecker />
  </React.StrictMode>
);
