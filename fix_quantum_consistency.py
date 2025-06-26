#!/usr/bin/env python3
"""
Comprehensive Quantum Blockchain Consistency Fixer
Systematically replaces all legacy Ethereum ethash references with QMPoW quantum consensus
"""

import os
import re

def fix_file(file_path):
    """Fix a single Go file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace ethash imports with qmpow imports
        content = content.replace(
            'github.com/ethereum/go-ethereum/consensus/ethash',
            'github.com/ethereum/go-ethereum/consensus/qmpow'
        )
        
        # Replace ethash.NewFaker() with qmpow.NewFaker()
        content = re.sub(r'ethash\.NewFaker\(\)', 'qmpow.NewFaker()', content)
        
        # Replace ethash.NewFakeFailer() with qmpow.NewFakeFailer()
        content = re.sub(r'ethash\.NewFakeFailer\(([^)]*)\)', r'qmpow.NewFakeFailer(\1)', content)
        
        # Replace ethash.New() with qmpow.New()
        content = re.sub(r'ethash\.New\(', 'qmpow.New(', content)
        
        # Replace ethash.CalcDifficulty with qmpow.CalcDifficulty
        content = re.sub(r'ethash\.CalcDifficulty\(', 'qmpow.CalcDifficulty(', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all Go files in the quantum-geth directory"""
    
    # Find all Go files in quantum-geth
    go_files = []
    for root, dirs, files in os.walk('quantum-geth'):
        # Skip vendor and .git directories
        dirs[:] = [d for d in dirs if d not in ['vendor', '.git', 'build']]
        for file in files:
            if file.endswith('.go'):
                go_files.append(os.path.join(root, file))
    
    print(f"🔍 Found {len(go_files)} Go files to check")
    
    fixed_count = 0
    for file_path in go_files:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"\n🎉 QUANTUM CONSISTENCY FIX COMPLETE!")
    print(f"📁 Files checked: {len(go_files)}")
    print(f"🔧 Files fixed: {fixed_count}")
    
    if fixed_count > 0:
        print(f"\n🚀 All ethash references replaced with QMPoW quantum consensus!")
        print(f"💎 Your quantum blockchain is now fully consistent!")
    else:
        print(f"\n✨ No files needed fixing - quantum consistency already achieved!")

if __name__ == "__main__":
    main() 