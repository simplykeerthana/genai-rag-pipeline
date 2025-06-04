#!/usr/bin/env python3
"""
Fix cache permission issues
"""
import os
import sys
import shutil
import stat
from pathlib import Path

def fix_permissions(directory):
    """Fix permissions for all files in directory"""
    print(f"Fixing permissions in {directory}")
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return
    
    # Fix directory permissions
    try:
        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        print(f"Fixed permissions for directory: {directory}")
    except Exception as e:
        print(f"Could not fix directory permissions: {e}")
    
    # Fix file permissions
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
            except Exception as e:
                print(f"Could not fix permissions for {dir_path}: {e}")
        
        for f in files:
            file_path = os.path.join(root, f)
            try:
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
                print(f"Fixed permissions for: {file_path}")
            except Exception as e:
                print(f"Could not fix permissions for {file_path}: {e}")

def clear_cache_forcefully(cache_dir="app/vector_cache"):
    """Clear cache by removing and recreating directory"""
    print(f"Forcefully clearing cache directory: {cache_dir}")
    
    if os.path.exists(cache_dir):
        try:
            # Try normal remove
            shutil.rmtree(cache_dir)
            print(f"Successfully removed {cache_dir}")
        except PermissionError:
            print("Permission error, trying alternative approach...")
            
            # Try to rename instead
            import time
            backup_name = f"{cache_dir}_backup_{int(time.time())}"
            try:
                os.rename(cache_dir, backup_name)
                print(f"Renamed {cache_dir} to {backup_name}")
            except Exception as e:
                print(f"Could not rename: {e}")
                
                # Last resort - delete individual files
                for root, dirs, files in os.walk(cache_dir):
                    for f in files:
                        file_path = os.path.join(root, f)
                        try:
                            os.remove(file_path)
                            print(f"Removed {file_path}")
                        except Exception as e:
                            print(f"Could not remove {file_path}: {e}")
    
    # Recreate directory
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Created fresh cache directory: {cache_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix cache permissions")
    parser.add_argument('--fix', action='store_true', help='Fix permissions')
    parser.add_argument('--clear', action='store_true', help='Clear cache forcefully')
    parser.add_argument('--dir', default='app/vector_cache', help='Cache directory')
    
    args = parser.parse_args()
    
    if args.fix:
        fix_permissions(args.dir)
    elif args.clear:
        clear_cache_forcefully(args.dir)
    else:
        print("Use --fix to fix permissions or --clear to clear cache")

if __name__ == "__main__":
    main()