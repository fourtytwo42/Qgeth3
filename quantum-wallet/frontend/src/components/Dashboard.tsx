import React, { useState } from 'react';
import { useWallet } from '../contexts/WalletContext';
import { 
  WalletIcon, 
  PlusIcon, 
  ArrowUpIcon, 
  ArrowDownIcon,
  CpuChipIcon,
  SignalIcon,
  GlobeAltIcon,
  CubeIcon,
  UsersIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';

export function Dashboard() {
  const {
    selectedAccount,
    accounts,
    networkInfo,
    isConnected,
    isLoading,
    createAccount,
    selectAccount
  } = useWallet();
  
  const [showCreateAccount, setShowCreateAccount] = useState(false);
  const [passphrase, setPassphrase] = useState('');

  const handleCreateAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!passphrase.trim()) return;
    
    try {
      await createAccount(passphrase);
      setPassphrase('');
      setShowCreateAccount(false);
    } catch (error) {
      console.error('Failed to create account:', error);
    }
  };

  const formatBalance = (balance: string) => {
    return parseFloat(balance).toFixed(4);
  };

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-quantum-950 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-quantum-500 to-neon-purple rounded-full flex items-center justify-center">
              <WalletIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                Quantum Wallet
              </h1>
              <p className="text-gray-400">Your gateway to the quantum blockchain</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              isConnected 
                ? 'bg-green-900/20 text-green-400 border border-green-400/20' 
                : 'bg-red-900/20 text-red-400 border border-red-400/20'
            }`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
            
            {networkInfo && (
              <div className="text-sm text-gray-400">
                Chain ID: {networkInfo.chainId}
              </div>
            )}
          </div>
        </motion.div>

        {/* Main Balance Card */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="quantum-card relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-quantum-600/10 to-neon-purple/10"></div>
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-6">
              <div>
                <p className="text-gray-400 mb-2">Total Balance</p>
                <div className="text-4xl font-bold text-white">
                  {selectedAccount ? formatBalance(selectedAccount.balance) : '0.0000'} 
                  <span className="text-xl text-quantum-400 ml-2">Q</span>
                </div>
              </div>
              
              <div className="flex space-x-3">
                <button className="quantum-button flex items-center space-x-2">
                  <ArrowUpIcon className="w-5 h-5" />
                  <span>Send</span>
                </button>
                <button className="quantum-button flex items-center space-x-2">
                  <ArrowDownIcon className="w-5 h-5" />
                  <span>Receive</span>
                </button>
              </div>
            </div>
            
            {/* Account Selector */}
            {accounts.length > 0 && (
              <div className="flex items-center space-x-3">
                <p className="text-gray-400 text-sm">Account:</p>
                <select
                  value={selectedAccount?.address || ''}
                  onChange={(e) => selectAccount(e.target.value)}
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm"
                >
                  {accounts.map((account) => (
                    <option key={account.address} value={account.address}>
                      {formatAddress(account.address)} - {formatBalance(account.balance)} Q
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Network Stats */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="quantum-card"
          >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <GlobeAltIcon className="w-5 h-5 mr-2 text-quantum-400" />
              Network Status
            </h3>
            
            {networkInfo ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Block Number</span>
                  <span className="text-white font-mono">{networkInfo.blockNumber.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Peers</span>
                  <span className="text-white flex items-center">
                    <UsersIcon className="w-4 h-4 mr-1" />
                    {networkInfo.peerCount}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Gas Price</span>
                  <span className="text-white font-mono">{parseInt(networkInfo.gasPrice).toLocaleString()} wei</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mining</span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    networkInfo.mining 
                      ? 'bg-green-900/20 text-green-400' 
                      : 'bg-gray-900/20 text-gray-400'
                  }`}>
                    {networkInfo.mining ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-gray-400 text-center py-4">
                No network connection
              </div>
            )}
          </motion.div>

          {/* Quick Actions */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="quantum-card"
          >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <CubeIcon className="w-5 h-5 mr-2 text-quantum-400" />
              Quick Actions
            </h3>
            
            <div className="space-y-3">
              <button 
                onClick={() => setShowCreateAccount(true)}
                className="w-full bg-gradient-to-r from-quantum-600/20 to-quantum-700/20 border border-quantum-500/30 rounded-lg p-3 text-left hover:border-quantum-400/50 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <PlusIcon className="w-5 h-5 text-quantum-400" />
                  <div>
                    <div className="text-white font-medium">Create Account</div>
                    <div className="text-gray-400 text-sm">Generate a new wallet address</div>
                  </div>
                </div>
              </button>
              
              <button className="w-full bg-gradient-to-r from-neon-purple/20 to-neon-pink/20 border border-neon-purple/30 rounded-lg p-3 text-left hover:border-neon-purple/50 transition-colors">
                <div className="flex items-center space-x-3">
                  <CpuChipIcon className="w-5 h-5 text-neon-purple" />
                  <div>
                    <div className="text-white font-medium">Start Mining</div>
                    <div className="text-gray-400 text-sm">Begin quantum mining operations</div>
                  </div>
                </div>
              </button>
              
              <button className="w-full bg-gradient-to-r from-neon-blue/20 to-neon-green/20 border border-neon-blue/30 rounded-lg p-3 text-left hover:border-neon-blue/50 transition-colors">
                <div className="flex items-center space-x-3">
                  <SignalIcon className="w-5 h-5 text-neon-blue" />
                  <div>
                    <div className="text-white font-medium">Console</div>
                    <div className="text-gray-400 text-sm">Access geth console</div>
                  </div>
                </div>
              </button>
            </div>
          </motion.div>

          {/* Recent Activity */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="quantum-card"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
            
            <div className="space-y-3">
              <div className="text-gray-400 text-center py-8">
                No recent transactions
              </div>
            </div>
          </motion.div>
        </div>

        {/* Create Account Modal */}
        {showCreateAccount && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowCreateAccount(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="quantum-card max-w-md w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-xl font-semibold text-white mb-4">Create New Account</h3>
              <form onSubmit={handleCreateAccount} className="space-y-4">
                <div>
                  <label className="block text-gray-400 text-sm mb-2">
                    Passphrase (keep this safe!)
                  </label>
                  <input
                    type="password"
                    value={passphrase}
                    onChange={(e) => setPassphrase(e.target.value)}
                    className="quantum-input w-full"
                    placeholder="Enter a secure passphrase"
                    required
                  />
                </div>
                <div className="flex space-x-3">
                  <button
                    type="button"
                    onClick={() => setShowCreateAccount(false)}
                    className="flex-1 bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isLoading || !passphrase.trim()}
                    className="flex-1 quantum-button"
                  >
                    {isLoading ? 'Creating...' : 'Create Account'}
                  </button>
                </div>
              </form>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
} 