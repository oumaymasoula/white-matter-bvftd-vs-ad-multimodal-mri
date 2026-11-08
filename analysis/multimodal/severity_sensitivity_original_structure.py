from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

root = Path.home() / "scratch/WM_Project/clinical_analysis"
out = root / "outputs/tables"
df = pd.read_csv(out / "clinical_imaging_307.tsv", sep="\t")
assert len(df) == 307
assert not df.duplicated(["dataset", "subject"]).any()

outcomes = [
    "frontal_composite_thickness_mm",
    "hippocampus_volume_ml",
    "log1p_WMH_ml",
    "ALPS_bilateral_mean_of_ratios",
]
variants = [
    "full_reference",
    "clinical_complete",
    "CDR_0p5_to_1",
    "CDR_and_MMSE_20_to_26",
    "CDR_MMSE_and_common_age",
]

for c in outcomes + ["age", "eTIV_ml", "MMSE", "CDR_global"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

counts, membership, estimates = [], [], []

for variant in variants:
    for dataset, disease in [("NIFD", "bvFTD"), ("ADNI", "AD")]:
        frame = df[
            (df.dataset == dataset) & df.group.isin(["CN", "MCI", "AD"] if dataset == "ADNI" else ["CN", "bvFTD"])
        ].copy()
        patient = frame.group.eq(disease)
        eligible = pd.Series(True, index=frame.index)

        if variant != "full_reference":
            complete = frame.MMSE.notna() & frame.CDR_global.notna()
            eligible &= ~patient | complete
        if variant in variants[2:]:
            eligible &= ~patient | frame.CDR_global.between(.5, 1)
        if variant in variants[3:]:
            eligible &= ~patient | frame.MMSE.between(20, 26)
        if variant == "CDR_MMSE_and_common_age":
            eligible &= frame.age.between(61.6905, 74.8024)

        frame = frame[eligible].copy()
        before = frame.groupby("group").size().to_dict()

        # Each retained site must support this patient-control comparison.
        support = pd.crosstab(frame.site, frame.group)
        if disease in support and "CN" in support:
            sites = support.index[
                (support[disease] > 0) & (support["CN"] > 0)
            ]
        else:
            sites = []
        frame = frame[frame.site.isin(sites)].copy()

        for group in ["CN", disease]:
            subset = frame[frame.group == group]
            counts.append(dict(
                variant=variant, dataset=dataset, group=group,
                n_before_site_support=before.get(group, 0),
                n_after_site_support=len(subset),
                age_mean=subset.age.mean(),
                age_min=subset.age.min(), age_max=subset.age.max(),
                MMSE_median=subset.MMSE.median(),
                CDR_median=subset.CDR_global.median()
            ))
        for subject in frame.subject:
            membership.append(dict(
                variant=variant, dataset=dataset, subject=subject
            ))

        for outcome in outcomes:
            result = dict(
                variant=variant, dataset=dataset, disease=disease,
                outcome=outcome, status="blocked", reason="",
                n_patient=0, n_control=0,
                beta=np.nan, se=np.nan, ci_low=np.nan,
                ci_high=np.nan, p=np.nan, max_leverage=np.nan,
                residual_df=np.nan, formula=""
            )
            columns = [outcome, "age", "sex", "site", "group"]
            acquisition = None
            if outcome == "ALPS_bilateral_mean_of_ratios":
                acquisition = (
                    "protocol" if dataset == "NIFD" else "scanner_model"
                )
                columns.append(acquisition)
            if outcome == "hippocampus_volume_ml":
                columns.append("eTIV_ml")

            sample = frame[columns].replace(
                [np.inf, -np.inf], np.nan
            ).dropna().copy()
            for c in ["sex", "site"] + ([acquisition] if acquisition else []):
                sample = sample[sample[c].astype(str).str.strip().ne("")]
            sample["patient"] = sample.group.eq(disease).astype(int)
            result["n_patient"] = int(sample.patient.sum())
            result["n_control"] = int(sample.group.eq("CN").sum())

            if min(result["n_patient"], result["n_control"]) < 10:
                result["reason"] = "fewer_than_10_in_one_group"
                estimates.append(result)
                continue

            sample["y"] = sample[outcome]
            sample["age_c"] = sample.age - sample.age.mean()
            terms = ["C(group, Treatment(reference='CN'))", "age_c"]
            coefficient = (
                f"C(group, Treatment(reference='CN'))[T.{disease}]"
            )
            for c in ["sex", "site"] + ([acquisition] if acquisition else []):
                if sample[c].nunique() > 1:
                    terms.append(f"C({c})")
            if outcome == "hippocampus_volume_ml":
                sample["etiv_c"] = sample.eTIV_ml - sample.eTIV_ml.mean()
                terms.append("etiv_c")
            formula = "y ~ " + " + ".join(terms)
            result["formula"] = formula

            try:
                ordinary = smf.ols(formula, data=sample).fit()
                design = ordinary.model.exog
                leverage = ordinary.get_influence().hat_matrix_diag
                result["max_leverage"] = float(leverage.max())
                result["residual_df"] = float(ordinary.df_resid)
                if np.linalg.matrix_rank(design) < design.shape[1]:
                    result["reason"] = "rank_deficient"
                elif ordinary.df_resid < 10:
                    result["reason"] = "fewer_than_10_residual_df"
                elif leverage.max() >= .99:
                    result["reason"] = "near_unit_leverage"
                else:
                    model = smf.ols(formula, data=sample).fit(
                        cov_type="HC3", use_t=True
                    )
                    low, high = model.conf_int().loc[coefficient]
                    result.update(
                        status="estimated",
                        beta=model.params[coefficient],
                        se=model.bse[coefficient],
                        ci_low=low, ci_high=high,
                        p=model.pvalues[coefficient]
                    )
            except Exception as error:
                result["reason"] = f"{type(error).__name__}: {error}"
            estimates.append(result)

est = pd.DataFrame(estimates)
cross = []
for variant in variants:
    for outcome in outcomes:
        rows = est[
            (est.variant == variant) & (est.outcome == outcome)
        ].set_index("dataset")
        row = dict(variant=variant, outcome=outcome,
                   status="blocked", difference=np.nan,
                   ci_low=np.nan, ci_high=np.nan, p=np.nan)
        if len(rows) == 2 and (rows.status == "estimated").all():
            difference = rows.loc["NIFD", "beta"] - rows.loc["ADNI", "beta"]
            se = np.sqrt(
                rows.loc["NIFD", "se"]**2 + rows.loc["ADNI", "se"]**2
            )
            row.update(
                status="estimated", difference=difference,
                ci_low=difference-1.96*se,
                ci_high=difference+1.96*se,
                p=2*norm.sf(abs(difference/se))
            )
        cross.append(row)

cross = pd.DataFrame(cross)
cross["q_targeted_4"] = np.nan
for variant in variants:
    mask = cross.variant.eq(variant)
    cross.loc[mask, "q_targeted_4"] = multipletests(
        cross.loc[mask, "p"].fillna(1), method="fdr_bh"
    )[1]
cross.loc[cross.status != "estimated", "q_targeted_4"] = np.nan

pd.DataFrame(counts).to_csv(
    out / "severity_variant_counts_original_structure.tsv", sep="\t", index=False)
pd.DataFrame(membership).to_csv(
    out / "severity_variant_membership_original_structure.tsv", sep="\t", index=False)
est.to_csv(
    out / "severity_within_cohort_effects_original_structure.tsv", sep="\t", index=False)
cross.to_csv(
    out / "severity_bvFTD_minus_AD_contrasts_original_structure.tsv", sep="\t", index=False)

print("\nSAMPLE COUNTS")
print(pd.DataFrame(counts).to_string(index=False))
print("\nCROSS-COHORT CONTRASTS")
print(cross.to_string(index=False))
print("\nMODEL DIAGNOSTICS")
print(est[["variant", "dataset", "outcome", "status",
           "n_patient", "n_control", "reason"]].to_string(index=False))
print("\nSaved:", out)
