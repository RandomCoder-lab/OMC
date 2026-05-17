"""Parity check — same task, same model, both runtimes.

Reports:
  PyTorch tail-mean loss
  Prometheus tail-mean loss (extracted from harmonic SGD run output)
  PyTorch argmax predictions
  Prometheus argmax predictions
  Final verdict: match / off-by-noise / divergent
"""

import re
import subprocess
import sys
from pathlib import Path


def run_pytorch():
    here = Path(__file__).parent
    r = subprocess.run(
        [sys.executable, str(here / "torch_baseline.py")],
        capture_output=True, text=True, check=True,
    )
    return r.stdout


def run_prometheus():
    """Use the harmonic SGD demo because its 'vanilla' arm is exactly
    the same SGD we want to compare, and it reports tail-mean."""
    root = Path(__file__).parent.parent.parent
    binary = root / "target" / "release" / "omnimcode-standalone"
    omc_file = root / "examples" / "prometheus_harmonic_sgd.omc"
    r = subprocess.run(
        [str(binary), str(omc_file)],
        capture_output=True, text=True, check=True,
        cwd=str(root),
    )
    return r.stdout


def extract_torch_loss(out):
    m = re.search(r"final tail-mean loss:\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None


def extract_prom_seed42_vanilla(out):
    # "seed 42  vanilla=0.02669096943022651  harmonic=..."
    m = re.search(r"seed 42\s+vanilla=([0-9.]+)", out)
    return float(m.group(1)) if m else None


def main():
    print("=== Prometheus ↔ PyTorch parity ===")
    print()
    print("Task: tinyLM, vocab=3 abc bigram, hidden=8, SGD lr=0.05,")
    print("      200 steps, seed=42, Xavier-uniform init, MSE loss")
    print("      Metric: mean loss over last 20 steps")
    print()

    torch_out = run_pytorch()
    print("--- PyTorch ---")
    print(torch_out.strip())
    print()

    prom_out = run_prometheus()
    print("--- Prometheus (vanilla SGD arm of harmonic A/B) ---")
    # Just show the seed 42 line.
    for line in prom_out.splitlines():
        if "seed 42" in line or "vanilla mean" in line or "harmonic mean" in line or "harmonic wins" in line:
            print(f"  {line}")
    print()

    torch_loss = extract_torch_loss(torch_out)
    prom_loss = extract_prom_seed42_vanilla(prom_out)
    if torch_loss is None or prom_loss is None:
        print("[ERROR] could not extract losses for comparison")
        sys.exit(1)

    delta = abs(torch_loss - prom_loss)
    rel = (delta / max(torch_loss, prom_loss)) * 100
    print(f"PyTorch  tail-mean: {torch_loss:.6f}")
    print(f"Prom     tail-mean: {prom_loss:.6f}")
    print(f"abs delta:          {delta:.6f}")
    print(f"rel delta:          {rel:.3f}%")
    print()

    if rel < 5:
        print("[PARITY] Prometheus matches PyTorch within <5% on identical")
        print("         task + architecture + seed. The substrate-native")
        print("         training loop is producing PyTorch-comparable results.")
    elif rel < 20:
        print("[CLOSE]  Prometheus tracks PyTorch within 20%. Reasonable")
        print("         given different numerical orderings; not bit-identical")
        print("         but architecturally equivalent.")
    else:
        print("[DIFF]   Numbers diverge significantly. Investigate: init,")
        print("         update order, gradient computation, or numerical")
        print("         precision differences between tape and torch.autograd.")


if __name__ == "__main__":
    main()
