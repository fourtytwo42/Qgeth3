import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

// Define types for wallet state
interface Account {
  address: string;
  balance: string;
}

interface Transaction {
  hash: string;
  from: string;
  to: string;
  value: string;
  timestamp: number;
  status: string;
}

interface NetworkInfo {
  chainId: number;
  networkName: string;
  blockNumber: number;
  peerCount: number;
  syncing: boolean;
  gasPrice: string;
  difficulty: string;
  hashRate: string;
}

interface WalletState {
  accounts: Account[];
  selectedAccount: Account | null;
  transactions: Transaction[];
  networkInfo: NetworkInfo | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
}

interface WalletContextType extends WalletState {
  // Account functions
  createAccount: (passphrase: string) => Promise<string>;
  selectAccount: (address: string) => void;
  refreshBalance: () => Promise<void>;
  
  // Transaction functions
  sendTransaction: (to: string, amount: string, passphrase: string) => Promise<string>;
  refreshTransactions: () => Promise<void>;
  
  // Mining functions
  startMining: () => Promise<void>;
  stopMining: () => Promise<void>;
  
  // Network functions
  connectToNode: (endpoint: string) => Promise<void>;
  refreshNetworkInfo: () => Promise<void>;
  
  // Console functions
  executeCommand: (command: string) => Promise<string>;
}

const WalletContext = createContext<WalletContextType | null>(null);

export function useWallet() {
  const context = useContext(WalletContext);
  if (!context) {
    throw new Error('useWallet must be used within a WalletProvider');
  }
  return context;
}

interface WalletProviderProps {
  children: ReactNode;
}

// Declare Wails backend functions
declare global {
  interface Window {
    backend: {
      WalletService: {
        CreateAccount: (passphrase: string) => Promise<string>;
        GetAccounts: () => Promise<Account[]>;
        GetBalance: (address: string) => Promise<string>;
        SendTransaction: (from: string, to: string, amount: string, passphrase: string) => Promise<string>;
        GetTransactions: (address: string) => Promise<Transaction[]>;
        StartMining: () => Promise<void>;
        StopMining: () => Promise<void>;
        ConnectToNode: (endpoint: string) => Promise<void>;
        GetNetworkInfo: () => Promise<NetworkInfo>;
        ExecuteConsoleCommand: (command: string) => Promise<string>;
      };
    };
  }
}

export function WalletProvider({ children }: WalletProviderProps) {
  const [state, setState] = useState<WalletState>({
    accounts: [],
    selectedAccount: null,
    transactions: [],
    networkInfo: null,
    isConnected: false,
    isLoading: false,
    error: null,
  });

  // Helper function to call backend API with proper error handling
  const callAPI = async (method: keyof typeof window.backend.WalletService, ...args: any[]): Promise<any> => {
    try {
      // Try direct Wails runtime calls for testing
      if (typeof window !== 'undefined' && (window as any).go) {
        const go = (window as any).go;
        console.log('Available Go methods:', Object.keys(go));
        
        // Try calling methods directly on the main app
        if (go.main && go.main.App) {
          console.log('App methods:', Object.keys(go.main.App));
          
          // Test the simple Greet method first
          if (method === 'GetAccounts' && go.main.App.Greet) {
            const greeting = await go.main.App.Greet('Test');
            console.log('Greet test result:', greeting);
          }
          
          // Try calling the actual method
          const appMethod = go.main.App[method];
          if (appMethod) {
            console.log(`Calling ${method} with args:`, args);
            return await appMethod(...args);
          }
        }
      }
      
      if (!window.backend?.WalletService) {
        // Fallback for development/testing - use mock data
        console.log(`Mock API call: ${method}`, args);
        switch (method) {
          case 'GetAccounts':
            return [
              { address: '0x742d35Cc6BfE4C1672C4C6B1e4d50e8B6B8B7B0f', balance: '125.5' },
              { address: '0x8ba1f109551bD432803012645Hac136c0A7c2B10', balance: '42.3' }
            ] as Account[];
          case 'GetNetworkInfo':
            return {
              chainId: 73235,
              networkName: 'Quantum Testnet',
              blockNumber: 15243,
              peerCount: 8,
              syncing: false,
              gasPrice: '20000000000',
              difficulty: '12345678',
              hashRate: '0'
            } as NetworkInfo;
          case 'GetTransactions':
            return [
              {
                hash: '0x1234567890abcdef...',
                from: '0x742d35Cc6BfE4C1672C4C6B1e4d50e8B6B8B7B0f',
                to: '0x8ba1f109551bD432803012645Hac136c0A7c2B10',
                value: '10.5',
                timestamp: Date.now() - 3600000,
                status: 'confirmed'
              }
            ] as Transaction[];
          case 'CreateAccount':
            return '0x' + Math.random().toString(16).substring(2, 42);
          case 'GetBalance':
            return '0.0';
          case 'SendTransaction':
            return '0x' + Math.random().toString(16).substring(2, 66);
          case 'StartMining':
          case 'StopMining':
          case 'ConnectToNode':
            return true;
          case 'ExecuteConsoleCommand':
            return 'Command executed successfully (mock)';
          default:
            throw new Error(`Unknown API method: ${method}`);
        }
      }

      // Real API call
      const apiMethod = window.backend.WalletService[method] as any;
      return await apiMethod(...args);
    } catch (error) {
      console.error(`API call failed: ${method}`, error);
      throw error;
    }
  };

  // Account functions
  const createAccount = async (passphrase: string): Promise<string> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      const address = await callAPI('CreateAccount', passphrase);
      await refreshAccounts();
      return address;
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
      throw error;
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const selectAccount = (address: string) => {
    const account = state.accounts.find(acc => acc.address === address);
    if (account) {
      setState(prev => ({ ...prev, selectedAccount: account }));
    }
  };

  const refreshAccounts = async () => {
    try {
      const accounts = await callAPI('GetAccounts');
      setState(prev => ({ 
        ...prev, 
        accounts, 
        selectedAccount: prev.selectedAccount || accounts[0] || null 
      }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    }
  };

  const refreshBalance = async () => {
    if (!state.selectedAccount) return;
    try {
      const balance = await callAPI('GetBalance', state.selectedAccount.address);
      setState(prev => ({
        ...prev,
        selectedAccount: prev.selectedAccount ? { ...prev.selectedAccount, balance } : null,
        accounts: prev.accounts.map(acc => 
          acc.address === state.selectedAccount?.address ? { ...acc, balance } : acc
        )
      }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    }
  };

  // Transaction functions
  const sendTransaction = async (to: string, amount: string, passphrase: string): Promise<string> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      const txHash = await callAPI('SendTransaction', state.selectedAccount?.address, to, amount, passphrase);
      await refreshBalance();
      await refreshTransactions();
      return txHash;
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
      throw error;
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const refreshTransactions = async () => {
    if (!state.selectedAccount) return;
    try {
      const transactions = await callAPI('GetTransactions', state.selectedAccount.address);
      setState(prev => ({ ...prev, transactions }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    }
  };

  // Mining functions
  const startMining = async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      await callAPI('StartMining');
      await refreshNetworkInfo();
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const stopMining = async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      await callAPI('StopMining');
      await refreshNetworkInfo();
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  // Network functions
  const connectToNode = async (endpoint: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      await callAPI('ConnectToNode', endpoint);
      setState(prev => ({ ...prev, isConnected: true }));
      await refreshNetworkInfo();
      await refreshAccounts();
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message, isConnected: false }));
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const refreshNetworkInfo = async () => {
    try {
      const networkInfo = await callAPI('GetNetworkInfo');
      setState(prev => ({ ...prev, networkInfo, isConnected: true }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message, isConnected: false }));
    }
  };

  // Console functions
  const executeCommand = async (command: string): Promise<string> => {
    try {
      return await callAPI('ExecuteConsoleCommand', command);
    } catch (error) {
      throw error;
    }
  };

  // Initialize wallet on mount
  useEffect(() => {
    const initializeWallet = async () => {
      setState(prev => ({ ...prev, isLoading: true }));
      try {
        // Try to connect to default node
        await connectToNode('http://localhost:8545');
      } catch (error) {
        console.log('Failed to connect to default node:', error);
        // Still load mock data for UI testing
        await refreshNetworkInfo();
        await refreshAccounts();
        setState(prev => ({ ...prev, isLoading: false }));
      }
    };

    initializeWallet();
  }, []);

  // Auto-refresh data every 30 seconds
  useEffect(() => {
    if (!state.isConnected) return;

    const interval = setInterval(async () => {
      await Promise.all([
        refreshNetworkInfo(),
        refreshBalance(),
        refreshTransactions(),
      ]);
    }, 30000);

    return () => clearInterval(interval);
  }, [state.isConnected, state.selectedAccount]);

  const contextValue: WalletContextType = {
    ...state,
    createAccount,
    selectAccount,
    refreshBalance,
    sendTransaction,
    refreshTransactions,
    startMining,
    stopMining,
    connectToNode,
    refreshNetworkInfo,
    executeCommand,
  };

  return (
    <WalletContext.Provider value={contextValue}>
      {children}
    </WalletContext.Provider>
  );
} 