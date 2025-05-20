import torch
import wandb
import os

def test_cuda():
    print("\n=== CUDA Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test a simple tensor operation
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("\nSimple CUDA operation test successful!")
    else:
        print("WARNING: CUDA is not available. Training will be very slow!")

def test_wandb():
    print("\n=== WandB Test ===")
    try:
        # Set API key (you can do this in three ways):
        # 1. Environment variable (recommended):
        # os.environ["WANDB_API_KEY"] = "your-api-key-here"
        
        # 2. Login programmatically (alternative):
        # wandb.login(key="your-api-key-here")
        
        # 3. Use wandb.login() without key to use the key from wandb login command
        
        # Initialize a test run
        run = wandb.init(
            project="GW_LLM-test",
            name="setup-test",
            config={
                "test_param": 42,
                "cuda_available": torch.cuda.is_available()
            }
        )
        
        # Log some test data
        wandb.log({"test_metric": 1.0})
        
        # Test artifact creation
        artifact = wandb.Artifact('test-artifact', type='dataset')
        artifact.add_file('test_setup.py')
        run.log_artifact(artifact)
        
        print("WandB test successful! Check your dashboard at:")
        print(f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}")
        
        # End the run
        run.finish()
    except Exception as e:
        print(f"WandB test failed: {str(e)}")
        print("Make sure you're logged in with: wandb login")
        print("Or set the API key using one of these methods:")
        print("1. Environment variable: export WANDB_API_KEY=your-key")
        print("2. Programmatic login: wandb.login(key='your-key')")
        print("3. Command line: wandb login")

if __name__ == "__main__":
    test_cuda()
    test_wandb() 