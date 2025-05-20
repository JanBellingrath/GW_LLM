import wandb
from sweep_config import create_sweep
import subprocess

if __name__ == '__main__':
    print("\n🚀 === Starting Sweep Runner === 🚀")
    try:
        # 1) create or reuse sweep on W&B
        print("🔄 Creating new W&B sweep configuration...")
        sweep_id = create_sweep()
        print(f"✅ Sweep created successfully! ID: {sweep_id}")
        
        # 2) launch a W&B agent via subprocess CLI, pointing to train.py as program
        cmd = f"wandb agent cerco_neuro_ai/GW_LLM/{sweep_id}"
        print(f"\n🤖 Launching W&B agent with command:")
        print(f"📝 {cmd}")
        
        print("\n⏳ Agent starting, please wait...")
        result = subprocess.run(cmd, shell=True, check=True)
        
        if result.returncode == 0:
            print("\n✅ Sweep completed successfully!")
        else:
            print(f"\n⚠️ Sweep finished with return code: {result.returncode}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running sweep agent: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {str(e)}")
        raise 