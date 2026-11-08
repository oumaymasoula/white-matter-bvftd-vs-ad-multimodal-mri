from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.home() / "scratch/WM_Project/clinical_analysis"
out = root / "outputs/tables"
df = pd.read_csv(out / "clinical_imaging_307.tsv", sep="\t")
assert len(df) == 307
assert not df.duplicated(["dataset", "subject"]).any()

# A fixed, small exploratory family defined before inspecting associations.
plan = []
for clinical in ["CDR_behavior", "NPI_apathy_severity",
                 "NPI_disinhibition_severity", "FAQ"]:
    plan.append(("NIFD", "bvFTD", clinical,
                 "frontal_composite_thickness_mm"))
for clinical in ["CDR_behavior", "FAQ"]:
    for imaging in ["log1p_WMH_ml", "ALPS_bilateral_mean_of_ratios"]:
        plan.append(("NIFD", "bvFTD", clinical, imaging))
for dataset, group in [("NIFD", "bvFTD"),
                       ("ADNI", "AD"), ("ADNI", "MCI")]:
    for imaging in ["frontal_composite_thickness_mm",
                    "hippocampus_volume_ml"]:
        plan.append((dataset, group, "MMSE", imaging))

results = []
for dataset, group, clinical, imaging in plan:
    row = dict(dataset=dataset, group=group, clinical=clinical,
               imaging=imaging, status="blocked", n=0,
               beta_standardized=np.nan, ci_low=np.nan,
               ci_high=np.nan, p=np.nan, max_leverage=np.nan,
               residual_df=np.nan, formula="", reason="")
    columns = [clinical, imaging, "age", "sex", "site"]
    if imaging == "hippocampus_volume_ml":
        columns.append("eTIV_ml")
    if any(c not in df.columns for c in columns):
        row["reason"] = "required_column_missing"
        results.append(row)
        continue

    sample = df[(df.dataset == dataset) & (df.group == group)][columns].copy()
    numeric = [clinical, imaging, "age"]
    if "eTIV_ml" in columns:
        numeric.append("eTIV_ml")
    for c in numeric:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    sample = sample[
        sample.sex.astype(str).str.strip().ne("") &
        sample.site.astype(str).str.strip().ne("")
    ].copy()

    row["n_before_site_restriction"] = len(sample)
    site_counts = sample.groupby("site").size()
    if dataset == "NIFD" and group == "bvFTD":
        # Choose using the full bvFTD imaging sample, before score missingness.
        full_counts = df[
            (df.dataset == "NIFD") & (df.group == "bvFTD")
        ].groupby("site").size()
        largest = full_counts.max()
        leaders = sorted(full_counts[full_counts == largest].index.astype(str))
        dominant_site = leaders[0]
        sample = sample[sample.site.astype(str) == dominant_site].copy()
        row["site_restriction"] = "dominant_bvFTD_site"
    else:
        retained_sites = site_counts[site_counts >= 3].index
        sample = sample[sample.site.isin(retained_sites)].copy()
        row["site_restriction"] = "at_least_3_complete_cases_per_site"
    row["retained_sites"] = "|".join(
        sorted(sample.site.astype(str).unique())
    )
    row["n"] = len(sample)

    if len(sample) < 15:
        row["reason"] = "fewer_than_15_complete_cases"
        results.append(row)
        continue
    if sample[clinical].nunique() < 3 or sample[imaging].std() == 0:
        row["reason"] = "insufficient_score_variation"
        results.append(row)
        continue

    sample["clinical_z"] = (
        sample[clinical] - sample[clinical].mean()
    ) / sample[clinical].std()
    sample["imaging_z"] = (
        sample[imaging] - sample[imaging].mean()
    ) / sample[imaging].std()
    sample["age_c"] = sample.age - sample.age.mean()
    terms = ["clinical_z", "age_c"]
    for c in ["sex", "site"]:
        if sample[c].nunique() > 1:
            terms.append(f"C({c})")
    if "eTIV_ml" in sample.columns:
        sample["etiv_c"] = sample.eTIV_ml - sample.eTIV_ml.mean()
        terms.append("etiv_c")
    formula = "imaging_z ~ " + " + ".join(terms)
    row["formula"] = formula

    try:
        ordinary = smf.ols(formula, data=sample).fit()
        design = ordinary.model.exog
        leverage = ordinary.get_influence().hat_matrix_diag
        row["max_leverage"] = float(leverage.max())
        row["residual_df"] = float(ordinary.df_resid)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            row["reason"] = "rank_deficient"
        elif ordinary.df_resid < 10:
            row["reason"] = "fewer_than_10_residual_degrees_of_freedom"
        elif leverage.max() >= .99:
            row["reason"] = "near_unit_leverage_HC3_unstable"
        else:
            robust = smf.ols(formula, data=sample).fit(
                cov_type="HC3", use_t=True
            )
            low, high = robust.conf_int().loc["clinical_z"]
            row.update(status="estimated",
                       beta_standardized=robust.params["clinical_z"],
                       ci_low=low, ci_high=high,
                       p=robust.pvalues["clinical_z"], reason="")
    except Exception as error:
        row["reason"] = f"{type(error).__name__}: {error}"
    results.append(row)

result = pd.DataFrame(results)
# Blocked planned tests remain in the correction family with p=1.
result["q_FDR"] = multipletests(
    result.p.fillna(1).to_numpy(), method="fdr_bh"
)[1]
result.loc[result.status != "estimated", "q_FDR"] = np.nan
result.to_csv(out / "clinical_imaging_associations_site_supported.tsv",
              sep="\t", index=False)

# Descriptive cross-dataset comparison, not a test of severity equivalence.
a = pd.to_numeric(df.loc[
    (df.dataset == "NIFD") & (df.group == "bvFTD"), "MMSE"
], errors="coerce").dropna()
b = pd.to_numeric(df.loc[
    (df.dataset == "ADNI") & (df.group == "AD"), "MMSE"
], errors="coerce").dropna()
test = ttest_ind(a, b, equal_var=False)
comparison = pd.DataFrame([dict(
    comparison="bvFTD_minus_AD_MMSE",
    analysis="unadjusted_Welch_descriptive_cross_dataset",
    n_bvFTD=len(a), n_AD=len(b),
    mean_bvFTD=a.mean(), mean_AD=b.mean(),
    mean_difference=a.mean()-b.mean(),
    t=test.statistic, p=test.pvalue,
    interpretation="Does_not_establish_severity_equivalence"
)])
comparison.to_csv(out / "MMSE_bvFTD_AD_descriptive_comparison_site_run.tsv",
                  sep="\t", index=False)

estimated = result[result.status == "estimated"].copy()
if not estimated.empty:
    labels = [
        f"{r.group}: {r.clinical} / {r.imaging}"
        for r in estimated.itertuples()
    ]
    y = np.arange(len(estimated))
    fig, ax = plt.subplots(figsize=(12, max(4, .45*len(estimated))))
    beta = estimated.beta_standardized.to_numpy()
    ax.errorbar(beta, y,
                xerr=np.vstack([
                    beta-estimated.ci_low.to_numpy(),
                    estimated.ci_high.to_numpy()-beta
                ]),
                fmt="o", color="#245b78", capsize=3)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Imaging SD per clinical-score SD; 95% HC3 CI")
    ax.set_title("Exploratory clinical associations: restricted site samples")
    fig.tight_layout()
    fig.savefig(root / "outputs/figures/clinical_associations_site_supported.png", dpi=300)
    plt.close(fig)

print("\nASSOCIATION RESULTS")
print(result.to_string(index=False))
print("\nDESCRIPTIVE MMSE COMPARISON")
print(comparison.to_string(index=False))
print("\nSaved outputs to:", out)
