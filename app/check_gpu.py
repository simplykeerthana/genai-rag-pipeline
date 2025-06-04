#!/usr/bin/env python3
"""
Comprehensive GPU diagnostic script
"""
import subprocess
import platform
import sys

def check_system_gpu():
    """Check if system has GPU using various methods"""
    print("=" * 60)
    print("SYSTEM GPU CHECK")
    print("=" * 60)
    
    # Check OS
    os_type = platform.system()
    print(f"\nOperating System: {os_type}")
    print(f"Python Version: {sys.version}")
    
    if os_type == "Darwin":  # macOS
        print("\n🍎 macOS detected")
        try:
            # Check for Apple Silicon
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_info = result.stdout.strip()
            print(f"CPU: {cpu_info}")
            
            if "Apple" in cpu_info:
                print("\n✅ Apple Silicon detected (M1/M2/M3)")
                print("   - Has integrated GPU")
                print("   - PyTorch supports this via 'mps' device")
                print("\n   To use Apple Silicon GPU:")
                print("   1. Install PyTorch with MPS support:")
                print("      pip install torch torchvision torchaudio")
                print("   2. Use device='mps' instead of 'cuda'")
                
                # Check if MPS is available
                try:
                    import torch
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        print("\n   ✅ MPS (Metal Performance Shaders) is available!")
                        print("   You can use: device = torch.device('mps')")
                    else:
                        print("\n   ⚠️  MPS not available in current PyTorch")
                except:
                    pass
            else:
                print("\n⚠️  Intel Mac detected - No NVIDIA GPU support")
                print("   Intel Macs cannot run CUDA")
                
        except Exception as e:
            print(f"Error checking macOS hardware: {e}")
            
    elif os_type == "Linux" or os_type == "Windows":
        # Check for NVIDIA GPU
        print("\nChecking for NVIDIA GPU...")
        
        try:
            # Try nvidia-smi
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected!")
                print("\nGPU Information:")
                # Parse basic info
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'NVIDIA' in line and 'Driver' in line:
                        print(line.strip())
                    if '|' in line and '%' in line and 'MiB' in line:
                        print(line.strip())
            else:
                print("❌ No NVIDIA GPU detected (nvidia-smi not found)")
                
        except FileNotFoundError:
            print("❌ nvidia-smi not found - NVIDIA drivers not installed")
            print("\nTo install NVIDIA drivers:")
            if os_type == "Linux":
                print("   Ubuntu/Debian: sudo apt install nvidia-driver-xxx")
                print("   Or visit: https://www.nvidia.com/Download/index.aspx")
            else:
                print("   Visit: https://www.nvidia.com/Download/index.aspx")
        except Exception as e:
            print(f"Error checking for NVIDIA GPU: {e}")
            
    # Check current PyTorch installation
    print("\n" + "=" * 60)
    print("PYTORCH INSTALLATION CHECK")
    print("=" * 60)
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda if torch.version.cuda else 'None'}")
        
        # Check for other compute devices
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"MPS available: True (Apple Silicon GPU)")
            
        # Check build configuration
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_getDeviceCount'):
            print("PyTorch built with CUDA: Yes")
        else:
            print("PyTorch built with CUDA: No")
            
    except ImportError:
        print("❌ PyTorch not installed")

def get_install_recommendations():
    """Provide installation recommendations based on system"""
    print("\n" + "=" * 60)
    print("INSTALLATION RECOMMENDATIONS")
    print("=" * 60)
    
    os_type = platform.system()
    
    if os_type == "Darwin":  # macOS
        print("\nFor macOS:")
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            if "Apple" in result.stdout:
                print("\n1. For Apple Silicon (M1/M2/M3), install PyTorch with:")
                print("   pip install torch torchvision torchaudio")
                print("\n2. Then modify your code to use MPS:")
                print("   device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')")
            else:
                print("\n⚠️  Intel Macs cannot use CUDA")
                print("   Your only option is CPU computation")
        except:
            pass
            
    else:  # Linux/Windows
        print("\nFor NVIDIA GPU support:")
        print("\n1. First ensure NVIDIA drivers are installed:")
        print("   - Check with: nvidia-smi")
        print("   - Install from: https://www.nvidia.com/Download/index.aspx")
        
        print("\n2. Then install PyTorch with CUDA support:")
        print("   # For CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   # For CUDA 12.1:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n3. Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")

if __name__ == "__main__":
    check_system_gpu()
    get_install_recommendations()