import React from 'react';
import { WalletProvider } from './contexts/WalletContext';
import { Dashboard } from './components/Dashboard';
import './index.css';

function App() {
  return (
    <WalletProvider>
      <Dashboard />
    </WalletProvider>
  );
}

export default App; 