import subprocess
import sys
import time

def run_command(cmd):
    print(f"\n🚀 Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
    rc = process.poll()
    if rc != 0:
        print(f"❌ Command failed with exit code {rc}")
    else:
        print("✅ Command completed successfully!")
    return rc

def main():
    print("=== Active Learning Experiment Orchestrator ===")
    
    # 1. Baseline check / verification (completed by agent)
    
    # 2. Config A: ScaledExamplePrompterConfig with double_ended disabled
    print("\n--- KICKING OFF EXPERIMENT A: ScaledExample (Double Ended Disabled) ---")
    rc_a = run_command([
        "modal", "run", "main.py::main",
        "--prompter-type", "scaled_example",
        "--double-ended", "False"
    ])
    
    # 3. Config B: ScaledExamplePrompterConfig with discernability filter removed
    print("\n--- KICKING OFF EXPERIMENT B: ScaledExample (No Discernability Filter) ---")
    rc_b = run_command([
        "modal", "run", "main.py::main",
        "--prompter-type", "scaled_example",
        "--use-discernability", "False"
    ])
    
    # 4. Config C: IncreaseDifficultyPrompterConfig with target increase 25%
    print("\n--- KICKING OFF EXPERIMENT C: IncreaseDifficulty (Target 25%, 25 steps x 2 questions) ---")
    rc_c = run_command([
        "modal", "run", "main.py::main",
        "--prompter-type", "increase_difficulty",
        "--num-generation-steps", "25",
        "--questions-per-target", "2",
        "--delta-percent", "0.25"
    ])
    
    # 5. Config D: AddOptionPrompterConfig
    print("\n--- KICKING OFF EXPERIMENT D: AddOption (25 steps x 2 questions) ---")
    rc_d = run_command([
        "modal", "run", "main.py::main",
        "--prompter-type", "add_option",
        "--num-generation-steps", "25",
        "--questions-per-target", "2"
    ])
    
    print("\n=== All experiments finished! ===")

if __name__ == "__main__":
    main()
