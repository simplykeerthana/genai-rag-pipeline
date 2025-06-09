#!/usr/bin/env python3
"""
Download Mistral-7B-Instruct-v0.2 using transformers library
Saves to models/mistralai/ directory in project root
"""

import os
import sys
from pathlib import Path

def install_dependencies():
    """Install required packages if not available"""
    packages = ["transformers", "torch", "accelerate"]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    return True

def download_mistral_model():
    """Download Mistral model using transformers"""
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir if script_dir.name != 'scripts' else script_dir.parent
    model_save_path = project_root / "models" / "mistralai" / "Mistral-7B-Instruct-v0.2"
    
    print(f"📂 Project root: {project_root}")
    print(f"📂 Model will be saved to: {model_save_path}")
    
    # Check if model already exists
    if model_save_path.exists() and any(model_save_path.iterdir()):
        print(f"✅ Model directory already exists")
        
        # Check if it's complete
        config_file = model_save_path / "config.json"
        if config_file.exists():
            print("✅ Model appears to be complete")
            response = input("Re-download? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("🔄 Using existing model")
                return True
        
        print("🗑️  Removing existing incomplete model...")
        import shutil
        shutil.rmtree(model_save_path)
    
    # Create directory
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print(f"\n📥 Downloading {model_name}...")
    print("⏱️  This will take 10-30 minutes (~13.5GB)")
    print("🌐 Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
    
    try:
        print("🔄 Downloading tokenizer...")
        # Download tokenizer first (smaller, faster)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=None,
            force_download=False
        )
        
        print("💾 Saving tokenizer...")
        tokenizer.save_pretrained(model_save_path)
        print("✅ Tokenizer saved successfully")
        
        print("\n🔄 Downloading model (this is the big one)...")
        # Download model with auto device mapping and torch dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=None,
            force_download=False,
            low_cpu_mem_usage=True  # More memory efficient
        )
        
        print("💾 Saving model...")
        model.save_pretrained(model_save_path)
        print("✅ Model saved successfully")
        
        # Clear memory
        del model
        del tokenizer
        
        print(f"\n✅ {model_name} downloaded and saved successfully!")
        
        # Verify download
        verify_model(model_save_path)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check your internet connection")
        print("   - Make sure you have enough disk space (~15GB)")
        print("   - Make sure you have enough RAM (~16GB recommended)")
        print("   - Try running: pip install --upgrade transformers torch")
        return False

def verify_model(model_path):
    """Verify the downloaded model"""
    print("\n🔍 Verifying download...")
    
    # Check essential files
    essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file in essential_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        return False
    
    # Check for model weights
    weight_files = (
        list(model_path.glob("*.safetensors")) + 
        list(model_path.glob("*.bin"))
    )
    
    if not weight_files:
        print("⚠️  No model weight files found")
        return False
    
    # Calculate size
    try:
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"📊 Total size: {size_gb:.1f} GB")
    except Exception:
        print("📊 Could not calculate size")
    
    # List key files
    print(f"\n📁 Key files in {model_path.name}:")
    key_files = ["config.json", "tokenizer.json"] + [f.name for f in weight_files[:3]]
    for file in key_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
    
    if len(weight_files) > 3:
        print(f"   ... and {len(weight_files)-3} more weight files")
    
    print("✅ Model verification passed!")
    return True

def show_next_steps():
    """Show next steps after download"""
    print(f"\n🎉 Mistral model ready!")
    print("\n📋 Next steps:")
    print("1. Setup backend:")
    print("   cd backend")
    print("   python -m venv venv")
    print("   source venv/bin/activate")
    print("   pip install -r requirements.txt")
    print()
    print("2. Setup frontend:")
    print("   cd frontend")
    print("   npm install")
    print()
    print("3. Start the application:")
    print("   Backend:  uvicorn app.main:app --reload")
    print("   Frontend: npm run dev")
    print()
    print("4. Visit: http://localhost:3000")
    print()
    print("💡 The model is ready for 'transformers' mode in your RAG pipeline!")

def main():
    """Main function"""
    print("🚀 Mistral Model Downloader (Transformers)")
    print("=" * 50)
    
    try:
        if download_mistral_model():
            show_next_steps()
        else:
            print("\n❌ Download failed. Please try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        print("💡 You can resume by running this script again")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()