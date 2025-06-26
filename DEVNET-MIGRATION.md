# DevNet Directory Migration Guide

## Overview

Starting with this release, Q Coin DevNet now uses standard directory structures aligned with TestNet and MainNet:

- **Old DevNet Location**: `./qdata/` (project root)
- **New DevNet Location**: 
  - Windows: `%APPDATA%\Qcoin\devnet\`
  - Linux: `$HOME/.qcoin/devnet/`

## Why This Change?

1. **Consistency**: All networks now use the same directory pattern
2. **Standard Practice**: Follows Ethereum and other blockchain conventions  
3. **User Data Protection**: Blockchain data stored in user directories, not project root
4. **Multi-Network Support**: Clear separation between mainnet, testnet, and devnet

## Migration Steps

### Automatic Migration (Recommended)

The reset scripts will automatically use the new location:

```bash
# Linux
./dev-reset-blockchain.sh --yes

# Windows
.\dev-reset-blockchain.ps1 -force
```

### Manual Migration (Preserve Existing Data)

If you want to keep your existing DevNet blockchain data:

#### Windows PowerShell:
```powershell
# Create new directory
New-Item -ItemType Directory -Path "$env:APPDATA\Qcoin\devnet" -Force

# Copy existing data (if you want to preserve it)
if (Test-Path "qdata") {
    Copy-Item -Recurse -Force "qdata\*" "$env:APPDATA\Qcoin\devnet\"
    Write-Host "DevNet data migrated to: $env:APPDATA\Qcoin\devnet"
}

# Clean up old directory
Remove-Item -Recurse -Force "qdata" -ErrorAction SilentlyContinue
```

#### Linux Bash:
```bash
# Create new directory
mkdir -p "$HOME/.qcoin/devnet"

# Copy existing data (if you want to preserve it)
if [ -d "qdata" ]; then
    cp -r qdata/* "$HOME/.qcoin/devnet/"
    echo "DevNet data migrated to: $HOME/.qcoin/devnet"
fi

# Clean up old directory
rm -rf qdata
```

## Directory Structure Comparison

### Before (Old Structure):
```
Qgeth3/
├── qdata/                    # DevNet data (project root)
│   ├── geth/
│   ├── keystore/
│   └── ...
├── quantum-geth/
└── ...
```

### After (New Structure):
```
# Windows
%APPDATA%\Qcoin\
├── devnet/                   # DevNet data
│   ├── geth/
│   ├── keystore/
│   └── ...
├── (testnet data at root)    # TestNet data  
└── mainnet/                  # MainNet data

# Linux  
$HOME/.qcoin/
├── devnet/                   # DevNet data
│   ├── geth/
│   ├── keystore/
│   └── ...
├── (testnet data at root)    # TestNet data
└── mainnet/                  # MainNet data
```

## Updated Commands

### Starting DevNet Node:
```bash
# Linux
./start-geth.sh devnet

# Windows
.\start-geth.ps1 devnet
```

### DevNet Reset:
```bash
# Linux
./dev-reset-blockchain.sh --yes

# Windows  
.\dev-reset-blockchain.ps1 -force
```

### Attach to DevNet Console:
```bash
# Linux
./geth.bin attach ipc:$HOME/.qcoin/devnet/geth.ipc

# Windows
.\geth.exe attach ipc:\\.\pipe\geth.ipc
```

## Verification

After migration, verify the new location:

### Windows:
```powershell
# Check if new directory exists
Test-Path "$env:APPDATA\Qcoin\devnet"

# List contents
Get-ChildItem "$env:APPDATA\Qcoin\devnet"
```

### Linux:
```bash
# Check if new directory exists
ls -la "$HOME/.qcoin/devnet"

# Verify geth data
ls -la "$HOME/.qcoin/devnet/geth/"
```

## Troubleshooting

### "Directory not found" errors:
- Make sure you're using the updated startup scripts
- Check that the new directory was created properly
- Run the reset script to initialize the new location

### Missing keystore:
- Your keystores are now in the new devnet directory
- Import accounts using the new path: `$HOME/.qcoin/devnet/keystore/`

### Old qdata directory still exists:
- The old `qdata` directory can be safely deleted after migration
- It's now included in `.gitignore` as legacy

## Support

If you encounter issues during migration:

1. Try a fresh reset: `./dev-reset-blockchain.sh --yes`
2. Check that you have write permissions to the new directory
3. Ensure sufficient disk space in the user directory
4. Create an issue on GitHub with error details

The new directory structure provides better organization and follows blockchain best practices while maintaining full compatibility with your existing DevNet setup. 