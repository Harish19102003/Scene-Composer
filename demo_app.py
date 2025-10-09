"""
Demo script to show the Gradio app functionality
"""

import os
import sys

def main():
    print("🎨 Image-to-Scene Narration Gradio App Demo")
    print("=" * 50)
    
    # Check if model exists
    checkpoint_path = "checkpoints/model_best.pth.tar"
    vocab_path = "checkpoints/vocab.pkl"
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
        print("❌ Model files not found!")
        print("Please train a model first:")
        print("  python start_training.py")
        print("\nOr use the demo mode with a mock model...")
        
        # Create a simple demo without actual model
        create_demo_app()
    else:
        print("✅ Model files found!")
        print("Launching full app...")
        launch_full_app()

def create_demo_app():
    """Create a demo app without requiring trained model"""
    import gradio as gr
    import numpy as np
    from PIL import Image
    
    def mock_generate_narration(image, max_length, temperature, use_sampling):
        """Mock narration generation for demo"""
        if image is None:
            return "Please upload an image first.", None
        
        # Generate a mock narration based on image properties
        width, height = image.size
        
        mock_narrations = [
            f"This is a {width}x{height} image showing a beautiful scene with various objects and elements. The composition is well-balanced and contains interesting visual details that tell a story about the environment and subjects captured in the frame.",
            f"The image displays a scene with dimensions {width} by {height} pixels, featuring multiple elements that create a compelling visual narrative. The lighting and composition work together to highlight the main subjects and create depth in the scene.",
            f"A {width}x{height} image presenting a detailed scene with rich visual information. The various components in the frame interact to create an engaging visual story that captures the essence of the moment and environment."
        ]
        
        import random
        narration = random.choice(mock_narrations)
        
        # Create a mock attention visualization
        mock_attention = create_mock_attention_vis(image)
        
        return narration, mock_attention
    
    def create_mock_attention_vis(image):
        """Create a mock attention visualization"""
        try:
            import matplotlib.pyplot as plt
            
            # Create a simple attention map
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Mock attention map
            attention_map = np.random.rand(14, 14)
            im = axes[1].imshow(attention_map, cmap='hot', interpolation='nearest')
            axes[1].set_title('Mock Attention Map')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            plt.tight_layout()
            
            # Convert to numpy array
            fig.canvas.draw()
            vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return vis_array
            
        except Exception as e:
            print(f"Error creating mock attention: {e}")
            return None
    
    # Create demo interface
    with gr.Blocks(title="Image-to-Scene Narration Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ Image-to-Scene Narration Demo
        
        **This is a demo version without a trained model.**
        
        Upload an image to see how the interface works. The narration will be generated using mock data.
        To use the full model, please train it first using `python start_training.py`.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                
                with gr.Accordion("Generation Parameters", open=False):
                    max_length = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Maximum Length"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature"
                    )
                    
                    use_sampling = gr.Checkbox(
                        label="Use Sampling",
                        value=False
                    )
                
                generate_btn = gr.Button("Generate Narration", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Generated Narration",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True
                )
                
                attention_vis = gr.Image(
                    label="Attention Visualization",
                    height=400
                )
        
        generate_btn.click(
            fn=mock_generate_narration,
            inputs=[input_image, max_length, temperature, use_sampling],
            outputs=[output_text, attention_vis]
        )
    
    print("🚀 Launching demo app...")
    print("Go to: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)

def launch_full_app():
    """Launch the full app with trained model"""
    try:
        from app import main as app_main
        app_main()
    except Exception as e:
        print(f"Error launching full app: {e}")
        print("Falling back to demo mode...")
        create_demo_app()

if __name__ == "__main__":
    main()
