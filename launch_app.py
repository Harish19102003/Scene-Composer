"""
Simple launcher script for the Gradio app
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required files exist"""
    checkpoint_path = "checkpoints/model_best.pth.tar"
    vocab_path = "checkpoints/vocab.pkl"
    
    if not os.path.exists(checkpoint_path):
        print("❌ Model checkpoint not found!")
        print(f"Expected: {checkpoint_path}")
        print("\nPlease train a model first:")
        print("  python start_training.py")
        print("  or")
        print("  python train.py")
        return False
    
    if not os.path.exists(vocab_path):
        print("❌ Vocabulary file not found!")
        print(f"Expected: {vocab_path}")
        print("\nPlease train a model first:")
        print("  python start_training.py")
        print("  or")
        print("  python train.py")
        return False
    
    print("✅ Model files found!")
    return True

def install_gradio():
    """Install Gradio if not available"""
    try:
        import gradio
        print("✅ Gradio is already installed")
        return True
    except ImportError:
        print("📦 Installing Gradio...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=4.0.0"])
            print("✅ Gradio installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Gradio")
            print("Please install manually: pip install gradio")
            return False

def main():
    """Main launcher function"""
    print("🚀 Image-to-Scene Narration Gradio App Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Install Gradio if needed
    if not install_gradio():
        return
    
    # Launch the app
    print("\n🌐 Launching Gradio app...")
    print("The app will open in your browser automatically.")
    print("If it doesn't, go to: http://127.0.0.1:7860")
    print("\nPress Ctrl+C to stop the app.")
    print("=" * 50)
    
    try:
        # Import and run the app
        from app import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that the model files exist in the checkpoints/ directory")
        print("3. Try running: python app.py --checkpoint checkpoints/model_best.pth.tar --vocab checkpoints/vocab.pkl")

if __name__ == "__main__":
    main()
