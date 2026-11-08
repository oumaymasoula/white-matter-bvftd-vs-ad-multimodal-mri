"""Reproduce the manuscript's exploratory control-cohort FA comparison."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def yes(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "1.0", "yes", "pass"]
    )


def analyze(cohort, tracts):
    keys = ["dataset", "subject"]
    c = cohort.loc[yes(cohort.supported_site) & cohort.group.eq("CN"),
                   keys + ["age", "sex"]].copy()
    expected = {"ADNI": 76, "NIFD": 67}
    if c.groupby("dataset").size().to_dict() != expected or c.duplicated(keys).any():
        raise ValueError("Expected the original 143 unique site-supported controls.")
    c["age"] = pd.to_numeric(c.age, errors="raise")
    if not np.isfinite(c.age).all() or c.sex.isna().any():
        raise ValueError("Invalid control covariates.")
    ages = c.groupby("dataset").age.agg(["min", "max"])
    lo, hi = ages["min"].max(), ages["max"].min()
    c = c.loc[c.age.between(lo, hi)].copy()
    if c.groupby("dataset").size().to_dict() != {"ADNI": 64, "NIFD": 35}:
        raise ValueError("Age-overlap cohort does not match the manuscript check.")
    x = c.merge(tracts[keys + ["tract", "FA_mean", "primary_analysis_eligible"]],
                on=keys, how="left", validate="one_to_many")
    if (x.duplicated(keys + ["tract"]).any() or not x.groupby(keys).size().eq(20).all()
            or x.tract.nunique() != 20 or not yes(x.primary_analysis_eligible).all()):
        raise ValueError("Expected 20 unique eligible tracts per control.")
    x["FA_mean"] = pd.to_numeric(x.FA_mean, errors="raise")
    if not np.isfinite(x.FA_mean).all():
        raise ValueError("Nonfinite tract FA.")
    x["NIFD"] = x.dataset.eq("NIFD").astype(int)
    rows = []
    for tract, g in x.groupby("tract", sort=True):
        model = smf.ols("FA_mean ~ NIFD + age + C(sex)", data=g,
                        missing="raise").fit(cov_type="HC3", use_t=True)
        if np.linalg.matrix_rank(model.model.exog) != model.model.exog.shape[1]:
            raise ValueError(f"Rank-deficient design for {tract}")
        low, high = model.conf_int().loc["NIFD"]
        rows.append(dict(tract=tract, n_NIFD=int(g.NIFD.sum()),
                         n_ADNI=int((1-g.NIFD).sum()),
                         FA_NIFD_minus_ADNI=model.params["NIFD"],
                         CI_low=low, CI_high=high, p=model.pvalues["NIFD"]))
    result = pd.DataFrame(rows)
    if not np.isfinite(result[["p", "CI_low", "CI_high"]]).all().all():
        raise ValueError("Nonfinite control-cohort inference.")
    result["q_BH_20"] = multipletests(result.p, method="fdr_bh")[1]
    return result, (lo, hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "tracts", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output path.")
    result, (lo, hi) = analyze(pd.read_csv(args.cohort, sep="\t"),
                              pd.read_csv(args.tracts, sep="\t"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        result.to_csv(f, sep="\t", index=False)
    print(f"Shared age range: {lo:.2f}–{hi:.2f}; results: {args.out}")


if __name__ == "__main__":
    main()
