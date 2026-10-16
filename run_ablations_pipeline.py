# run_ablations_pipeline.py
import sys
import time
import subprocess
from evaluate_visualize import run_ablations_eval

ABLATIONS = [
    ("ablate_full", []),
    ("ablate_no_overlap", ["--w-overlap", "0.0"]),
    ("ablate_no_indiv", ["--indiv-weight", "0.0"]),
    ("ablate_no_col", ["--w-col", "0.0"]),
    ("ablate_high_dropout", ["--sensor-dropout", "0.25"]),
    ("ablate_no_ou", ["--ou-weight", "0.0"]),
]

def main():
    print("=" * 80)
    print("STARTING CONTROLLED ONE-FACTOR-AT-A-TIME ABLATION TRAINING (6 VARIANTS x 60k STEPS)")
    print("=" * 80)
    
    total_start = time.time()
    for idx, (exp_name, extra_args) in enumerate(ABLATIONS, 1):
        cmd = [
            sys.executable, "train.py",
            "--total-timesteps", "60000",
            "--rollout-steps", "2000",
            "--eval-interval", "5",
            "--batch-size", "512",
            "--seed", "42",
            "--exp-name", exp_name
        ] + extra_args
        
        print(f"\n[{idx}/6] Training {exp_name} with args: {extra_args}...")
        t0 = time.time()
        res = subprocess.run(cmd, check=True)
        dur = time.time() - t0
        print(f"[{idx}/6] Completed {exp_name} in {dur:.1f}s (Exit code: {res.returncode})")

    print("\n" + "=" * 80)
    print("ALL 6 ABLATION VARIANTS TRAINED. NOW EVALUATING ON 30 TEST EPISODES...")
    print("=" * 80)
    
    run_ablations_eval(episodes=30)
    print(f"\n[DONE] Full ablation pipeline finished in {(time.time() - total_start):.1f}s.")

if __name__ == "__main__":
    main()
