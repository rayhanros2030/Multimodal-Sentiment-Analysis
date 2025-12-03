"""
Script to explore CMU-MultimodalSDK structure and find CMU-MOSI labels
"""

import os
from pathlib import Path
import json

def explore_directory(root_dir):
    """Recursively explore directory and find relevant files"""
    
    root = Path(root_dir)
    
    print("="*80)
    print(f"Exploring: {root}")
    print("="*80)
    
    # List root contents
    print("\nRoot directory contents:")
    for item in sorted(root.iterdir()):
        if item.is_dir():
            print(f"  [DIR] {item.name}/")
        else:
            print(f"  [FILE] {item.name}")
    
    # Search for label-related files
    print("\n" + "="*80)
    print("Searching for label files...")
    print("="*80)
    
    label_patterns = ['*label*', '*sentiment*', '*.pkl', '*.json', '*MOSI*']
    
    for pattern in label_patterns:
        matches = list(root.rglob(pattern))
        if matches:
            print(f"\n{pattern}: {len(matches)} matches")
            for match in matches[:10]:
                print(f"  - {match.relative_to(root)}")
                if match.suffix in ['.json', '.txt'] and match.stat().st_size < 1000000:  # Less than 1MB
                    try:
                        with open(match, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(500)  # First 500 chars
                            print(f"    Preview: {content[:100]}...")
                    except:
                        pass
    
    # Look for Python files that might load labels
    print("\n" + "="*80)
    print("Python files that might load labels:")
    print("="*80)
    
    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if 'mosi' in content.lower() and ('label' in content.lower() or 'sentiment' in content.lower()):
                    print(f"\n  {py_file.relative_to(root)}")
                    # Print relevant lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines[:50], 1):
                        if 'mosi' in line.lower() or 'label' in line.lower() or 'sentiment' in line.lower():
                            print(f"    Line {i}: {line.strip()[:100]}")
        except Exception as e:
            pass
    
    # Look for CMU-MOSI data directories
    print("\n" + "="*80)
    print("Potential CMU-MOSI data directories:")
    print("="*80)
    
    for item in root.rglob("*"):
        if item.is_dir():
            dir_name = item.name.lower()
            if 'mosi' in dir_name or 'data' in dir_name:
                print(f"  [DIR] {item.relative_to(root)}")
                # List first few files
                try:
                    files = list(item.iterdir())[:5]
                    for f in files:
                        print(f"      - {f.name}")
                except:
                    pass

if __name__ == '__main__':
    import sys
    
    sdk_path = r"C:\Users\PC\Downloads\CMU-MultimodalSDK-extracted"
    
    if len(sys.argv) > 1:
        sdk_path = sys.argv[1]
    
    if not os.path.exists(sdk_path):
        print(f"ERROR: Path does not exist: {sdk_path}")
        sys.exit(1)
    
    explore_directory(sdk_path)

