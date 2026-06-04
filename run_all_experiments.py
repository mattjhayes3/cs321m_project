import subprocess
import sys
import argparse

# Definitions of all 13 single-round and 3 10-round configurations from final_manuscript.tex
EXPERIMENTS = {
    # --- SINGLE-ROUND EXPERIMENTS (1-13) ---
    1: {
        "name": "NearbyExample (No filtering)",
        "prompter_type": "nearby_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": False,
        "detailed_analysis_prompt": False,
    },
    2: {
        "name": "NearbyExample (No filtering, detailed prompt)",
        "prompter_type": "nearby_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": False,
        "detailed_analysis_prompt": True,
    },
    3: {
        "name": "NearbyExample (Discernment filtering)",
        "prompter_type": "nearby_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": True,
        "detailed_analysis_prompt": False,
    },
    4: {
        "name": "ScaledExample (Double-ended, basic prompt)",
        "prompter_type": "scaled_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "openai/gpt-5.5",
        "double_ended": True,
        "use_discernability": False,
        "detailed_analysis_prompt": False,
    },
    5: {
        "name": "ScaledExample (Double-ended, detailed prompt)",
        "prompter_type": "scaled_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "openai/gpt-5.5",
        "double_ended": True,
        "use_discernability": False,
        "detailed_analysis_prompt": True,
    },
    6: {
        "name": "ScaledExample (Single-ended, detailed prompt)",
        "prompter_type": "scaled_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "openai/gpt-5.5",
        "double_ended": False,
        "use_discernability": False,
        "detailed_analysis_prompt": True,
    },
    7: {
        "name": "ScaledExample (Double-ended, detailed prompt, Opus 4.7)",
        "prompter_type": "scaled_example",
        "max_rounds": 1,
        "num_generation_steps": 10,
        "questions_per_target": 5,
        "generator_model": "anthropic/claude-opus-4.7",
        "double_ended": True,
        "use_discernability": False,
        "detailed_analysis_prompt": True,
    },
    8: {
        "name": "IncreaseDifficulty ('Increase difficulty by 25%')",
        "prompter_type": "increase_difficulty",
        "max_rounds": 1,
        "num_generation_steps": 25,
        "questions_per_target": 2,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": False,
        "delta_percent": 0.25,
        "difficulty_multiplier": 1.25,
    },
    9: {
        "name": "IncreaseDifficulty ('2 times more difficult')",
        "prompter_type": "increase_difficulty",
        "max_rounds": 1,
        "num_generation_steps": 25,
        "questions_per_target": 2,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": False,
        "delta_percent": 0.25,
        "difficulty_multiplier": 2.0,
    },
    10: {
        "name": "IncreaseDifficulty ('3 times more difficult')",
        "prompter_type": "increase_difficulty",
        "max_rounds": 1,
        "num_generation_steps": 25,
        "questions_per_target": 2,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": False,
        "delta_percent": 0.25,
        "difficulty_multiplier": 3.0,
    },
    11: {
        "name": "AddOption (No offset)",
        "prompter_type": "add_option",
        "max_rounds": 1,
        "num_generation_steps": 50,
        "questions_per_target": 1,
        "generator_model": "openai/gpt-5.4",
        "use_discernability": False,
        "selector_offset": 0.0,
    },
    12: {
        "name": "AddOption (Offset -0.2)",
        "prompter_type": "add_option",
        "max_rounds": 1,
        "num_generation_steps": 50,
        "questions_per_target": 1,
        "generator_model": "openai/gpt-5.4",
        "use_discernability": False,
        "selector_offset": -0.2,
    },
    13: {
        "name": "AddOption (Offset -0.2, Discernment filtering)",
        "prompter_type": "add_option",
        "max_rounds": 1,
        "num_generation_steps": 50,
        "questions_per_target": 1,
        "generator_model": "openai/gpt-5.4",
        "use_discernability": True,
        "selector_offset": -0.2,
    },
    # --- 10-ROUND EXPERIMENTS (14-16) ---
    14: {
        "name": "10-Round IncreaseDifficulty (3x, Discernment)",
        "prompter_type": "increase_difficulty",
        "max_rounds": 10,
        "num_generation_steps": 25,
        "questions_per_target": 2,
        "generator_model": "openai/gpt-5.5",
        "use_discernability": True,
        "difficulty_multiplier": 3.0,
        "detailed_analysis_prompt": True,
    },
    15: {
        "name": "10-Round AddOption (No discernment, Offset 0.0)",
        "prompter_type": "add_option",
        "max_rounds": 10,
        "num_generation_steps": 50,
        "questions_per_target": 1,
        "generator_model": "openai/gpt-5.4",
        "use_discernability": False,
        "selector_offset": 0.0,
    },
    16: {
        "name": "10-Round AddOption (No discernment, Offset -0.2)",
        "prompter_type": "add_option",
        "max_rounds": 10,
        "num_generation_steps": 50,
        "questions_per_target": 1,
        "generator_model": "openai/gpt-5.4",
        "use_discernability": False,
        "selector_offset": -0.2,
    }
}

def run_command(cmd):
    print(f"\n🚀 Running command: {' '.join(cmd)}")
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

def build_modal_cmd(config_dict, seed=42):
    cmd = ["modal", "run", "main.py::main"]
    cmd.extend(["--seed", str(seed)])
    for key, val in config_dict.items():
        if key == "name":
            continue
        
        # Convert param key to CLI format (e.g. prompter_type -> --prompter-type)
        cli_flag = f"--{key.replace('_', '-')}"
        
        # Standardize boolean values to capitalized string representations
        if isinstance(val, bool):
            cmd.extend([cli_flag, str(val)])
        else:
            cmd.extend([cli_flag, str(val)])
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Active Learning Experiment Orchestrator")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    parser.add_argument("--run", type=str, help="Comma-separated list of experiment IDs to run (e.g. '1,5,16' or 'all')")
    parser.add_argument("--seed", type=int, default=42, help="Seed value (default: 42)")
    
    args = parser.parse_args()

    if args.list:
        print("=== Available Configurations (from final_manuscript.tex) ===")
        print("\n--- Single-Round (1 Round, 50 questions) ---")
        for i in range(1, 14):
            print(f"  [{i:2}] {EXPERIMENTS[i]['name']}")
        print("\n--- 10-Round (10 Rounds, 500 questions) ---")
        for i in range(14, 17):
            print(f"  [{i:2}] {EXPERIMENTS[i]['name']}")
        return

    if not args.run:
        parser.print_help()
        return

    to_run = []
    if args.run.lower() == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        try:
            to_run = [int(x.strip()) for x in args.run.split(",") if x.strip()]
        except ValueError:
            print("❌ Invalid argument format for --run. Use comma-separated numbers (e.g., '1,5,16') or 'all'.")
            sys.exit(1)

    # Validate IDs
    for idx in to_run:
        if idx not in EXPERIMENTS:
            print(f"❌ Unknown experiment ID: {idx}. Run with --list to see available IDs.")
            sys.exit(1)

    print(f"=== Starting Active Learning Experiment Suite (Running {len(to_run)} configs) ===")
    
    results = {}
    for idx in to_run:
        config = EXPERIMENTS[idx]
        print(f"\n==================================================")
        print(f"👉 EXPERIMENT {idx}: {config['name']}")
        print(f"==================================================")
        cmd = build_modal_cmd(config, seed=args.seed)
        rc = run_command(cmd)
        results[idx] = {
            "name": config["name"],
            "status": "SUCCESS" if rc == 0 else f"FAILED (exit code {rc})"
        }

    print("\n==================================================")
    print("=== SUMMARY OF RUNS ===")
    print("==================================================")
    for idx, res in results.items():
        print(f"Experiment {idx:2} ({res['name']}): {res['status']}")

if __name__ == "__main__":
    main()
