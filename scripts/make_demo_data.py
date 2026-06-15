from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic demo dataset for NetRepro."
    )
    parser.add_argument("--output-dir", type=str, default="demo_data")
    parser.add_argument("--num-genes", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layout",
        choices=["genes_by_samples", "samples_by_genes"],
        default="samples_by_genes",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    genes = [f"GENE{i}" for i in range(args.num_genes)]
    samples = [f"S{i}" for i in range(args.num_samples)]
    base = rng.normal(size=(args.num_samples, args.num_genes))

    states = {
        "tissue_normal_train": base,
        "tissue_cancer_train": base + rng.normal(0.6, 0.2, size=base.shape),
        "cell_dmso_train": base + rng.normal(0.2, 0.2, size=base.shape),
        "cell_treated_train": base + rng.normal(-0.4, 0.2, size=base.shape),
        "tissue_normal_val": base + rng.normal(0.0, 0.05, size=base.shape),
        "tissue_cancer_val": base + rng.normal(0.6, 0.2, size=base.shape),
        "cell_dmso_val": base + rng.normal(0.2, 0.2, size=base.shape),
        "cell_treated_val": base + rng.normal(-0.4, 0.2, size=base.shape),
    }

    for name, array in states.items():
        if args.layout == "samples_by_genes":
            df = pd.DataFrame(array, index=samples, columns=genes)
        else:
            df = pd.DataFrame(array.T, index=genes, columns=samples)
        df.to_csv(output_dir / f"{name}.csv")

    fingerprint_ids = [f"cmp_{i}" for i in range(8)]
    fingerprints = pd.DataFrame(
        rng.integers(0, 2, size=(len(fingerprint_ids), 2048)),
        index=fingerprint_ids,
    )
    fingerprints.to_csv(output_dir / "compound_fingerprints.csv")

    treated_dir = output_dir / "treated_compounds"
    treated_dir.mkdir(exist_ok=True)
    for compound_id in fingerprint_ids:
        treated = base + rng.normal(-0.3, 0.25, size=base.shape)
        if args.layout == "samples_by_genes":
            df = pd.DataFrame(treated, index=samples, columns=genes)
        else:
            df = pd.DataFrame(treated.T, index=genes, columns=samples)
        df.to_csv(treated_dir / f"{compound_id}.csv")

    print(f"Demo data written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
