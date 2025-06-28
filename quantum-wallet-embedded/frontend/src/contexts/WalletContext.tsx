import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import * as App from '../../wailsjs/go/main/App';

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
  chainId: string;
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

// Wails v2 backend functions are imported from generated bindings

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
  const callAPI = async (method: string, ...args: any[]): Promise<any> => {
    try {
      console.log(`Calling API method: ${method}`, args);
      
      // Map method names to actual App functions
      switch (method) {
        case 'GetAccounts':
          return await App.GetAccounts();
        case 'GetNetworkInfo':
          return await App.GetNetworkInfo();
        case 'GetTransactions':
          return await App.GetTransactions(args[0]);
        case 'CreateAccount':
          return await App.CreateAccount(args[0]);
        case 'GetBalance':
          return await App.GetBalance(args[0]);
        case 'SendTransaction':
          return await App.SendTransaction(args[0], args[1], args[2], args[3]);
        case 'StartMining':
          return await App.StartMining();
        case 'StopMining':
          return await App.StopMining();
        case 'ConnectToNode':
          return await App.ConnectToNode(args[0]);
        case 'ExecuteConsoleCommand':
          return await App.ExecuteConsoleCommand(args[0]);
        default:
          throw new Error(`Unknown API method: ${method}`);
      }
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
      console.log('Calling GetNetworkInfo API...');
      const networkInfo = await callAPI('GetNetworkInfo');
      console.log('Network info received:', networkInfo);
      
      // Check if we're actually connected based on network data
      const isConnected = networkInfo && 
                         networkInfo.chainId !== "0" && 
                         networkInfo.networkName !== "Disconnected";
      
      console.log('Connection status:', isConnected);
      setState(prev => ({ 
        ...prev, 
        networkInfo, 
        isConnected, 
        error: isConnected ? null : 'Q Geth node not connected'
      }));
    } catch (error) {
      console.error('GetNetworkInfo failed:', error);
      setState(prev => ({ 
        ...prev, 
        error: `Network API call failed: ${(error as Error).message}`, 
        isConnected: false,
        networkInfo: null
      }));
      // Still throw the error so initialization can handle it
      throw error;
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
      
      // Add a small delay to ensure backend is ready
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      try {
        console.log('Attempting to refresh network info...');
        await refreshNetworkInfo();
        console.log('Network info refreshed, loading accounts...');
        await refreshAccounts();
        console.log('Wallet initialization completed');
      } catch (error) {
        console.error('Failed to initialize wallet:', error);
        setState(prev => ({ 
          ...prev, 
          error: `Failed to connect to Q Geth node: ${error}`,
          isConnected: false 
        }));
      } finally {
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