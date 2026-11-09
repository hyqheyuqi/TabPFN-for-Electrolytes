"""Verify selected manuscript statistics using only the published result tables."""

from pathlib import Path
import math

import pandas as pd

ROOT = Path(__file__).resolve().parent


def ampere_mape():
    errors = pd.read_csv(
        ROOT / "results/ampere_conductivity/figure_source_data/Fig05_ape_distribution_source_data.csv"
    )
    assert not errors.duplicated(["seed", "row_index"]).any()
    assert errors.groupby("seed").size().tolist() == [60] * 6
    by_seed = errors.groupby("seed").absolute_percentage_error.mean()
    summary = pd.read_csv(ROOT / "results/ampere_conductivity/tables/fixed_metrics_summary.csv")
    saved = summary.set_index("metric").loc["MAPE"]
    assert math.isclose(by_seed.mean(), saved["mean"], abs_tol=1e-10)
    assert math.isclose(by_seed.std(), saved["std"], abs_tol=1e-10)
    return by_seed.mean(), by_seed.std()


def ampere_search_cost():
    results = pd.read_csv(ROOT / "results/representation_ablation/tables/overall_summary.csv")
    by_temperature = results.pivot(index="temperature_c", columns="representation", values="batch_cost_mean")
    assert by_temperature.index.tolist() == [25, 45]
    mean_cost = by_temperature.mean()
    proposed = mean_cost["Proposed 2D"]
    return proposed, 100 * (1 - proposed / mean_cost["Random"]), 100 * (1 - proposed / mean_cost["Composition"])


def clio_validation_cost():
    summary = pd.read_csv(ROOT / "objective_search/clio/acquisition_results/tables/summary.csv")
    validation = summary[summary.phase.eq("validation")].set_index(["model", "rule"])
    return validation.loc[("TabPFN", "Mean"), "cost_1.0"]


if __name__ == "__main__":
    mean, sd = ampere_mape()
    cost, random_reduction, composition_reduction = ampere_search_cost()
    print(f"AMPERE fixed-split MAPE: {mean:.4f}% (seed SD {sd:.4f}%)")
    print(f"AMPERE proposed 2D: {cost:.2f} evaluations; reductions {random_reduction:.1f}% vs random, "
          f"{composition_reduction:.1f}% vs composition")
    print(f"Clio TabPFN validation: {clio_validation_cost():.1f} evaluations")
