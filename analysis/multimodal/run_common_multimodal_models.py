import os
import math
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

OUT = Path(os.environ["OUT"])
KEY = ["dataset", "subject"]

def load(name):
    path = Path(os.environ[name])
    frame = pd.read_csv(path, sep="\t")
    for col in KEY + ["group"]:
        if col in frame:
            frame[col] = frame[col].astype(str).str.strip()
    if "group" in frame:
        frame["group"] = frame["group"].replace({"Probable_AD": "AD"})
    return frame

def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label}: missing columns {sorted(missing)}")

def unique(frame, columns, label):
    if frame.duplicated(columns).any():
        bad = frame.loc[frame.duplicated(columns, keep=False), columns]
        raise RuntimeError(f"{label}: duplicate keys\n{bad.to_string(index=False)}")

def save(frame, name):
    frame.to_csv(OUT / "tables" / f"{name}.tsv", sep="\t", index=False)
    frame.to_csv(OUT / "tables" / f"{name}.csv", index=False)

def choose(frame, alternatives, label):
    # Explicit priority, recorded in the provenance table.
    for col in alternatives:
        if col in frame:
            return col
    raise RuntimeError(f"No {label} column among {alternatives}")

def check_categories(series, label):
    if series.isna().any():
        raise RuntimeError(f"{label}: missing categorical values")
    values = series.astype(str).str.strip()
    invalid = values.str.lower().isin(
        ["", "nan", "none", "unknown", "missing", "na", "n/a", "null"]
    )
    if invalid.any():
        raise RuntimeError(f"{label}: unresolved categorical values")
    return values

def fdr(frame, grouping):
    frame = frame.copy()
    frame["BH_q"] = np.nan
    frame["FDR_family_complete"] = False
    for _, indices in frame.groupby(grouping, dropna=False).groups.items():
        p = pd.to_numeric(frame.loc[indices, "p_value"], errors="coerce")
        # Do not silently shrink the test family when models fail.
        if np.isfinite(p).all():
            frame.loc[indices, "BH_q"] = multipletests(
                p.to_numpy(), method="fdr_bh"
            )[1]
            frame.loc[indices, "FDR_family_complete"] = True
    return frame

print("===== LOADING AND VERIFYING COMMON SUBJECTS =====")

candidate = load("CANDIDATES")
alps = load("ALPS")
wmh = load("WMH")
dti = load("DTI")
fs = load("FS")

for name, frame in [
    ("candidate", candidate), ("ALPS", alps), ("WMH", wmh), ("FS", fs)
]:
    require(frame, KEY + ["group"], name)
    unique(frame, KEY, name)

require(dti, KEY + ["group", "tract", "primary_analysis_eligible"], "DTI")
unique(dti, KEY + ["tract"], "DTI")

expected = {
    ("ADNI", "CN"): 79,
    ("ADNI", "MCI"): 98,
    ("ADNI", "AD"): 48,
    ("NIFD", "CN"): 67,
    ("NIFD", "bvFTD"): 45,
}
observed = candidate.groupby(["dataset", "group"]).size().to_dict()
if observed != expected:
    raise RuntimeError(f"Unexpected candidate counts: {observed}")

# Candidate identity only; reload measurements from original sources.
base = candidate[KEY + ["group"]].copy()

provenance = []
for source_name, source in [("ALPS", alps), ("WMH", wmh), ("FS", fs)]:
    check = base.merge(
        source[KEY + ["group"]], on=KEY, how="left",
        suffixes=("", "_source"), validate="one_to_one", indicator=True
    )
    if not check["_merge"].eq("both").all():
        raise RuntimeError(f"Missing {source_name} participant")
    if not check["group"].eq(check["group_source"]).all():
        raise RuntimeError(f"{source_name}: diagnostic-label conflict")

# Covariates use the same documented metadata source in both analyses.
age_col = choose(wmh, ["age", "age_at_DWI_years"], "age")
site_col = choose(wmh, ["site", "acquisition_site"], "site")
require(wmh, ["sex", "scanner_model", "wmh_volume_ml"], "WMH metadata")
protocol_col = choose(
    alps, ["series_description", "protocol"], "DWI protocol"
)

meta = wmh[KEY + [age_col, site_col, "sex", "scanner_model",
                 "wmh_volume_ml"]].rename(
    columns={age_col: "age", site_col: "site"}
)
base = base.merge(meta, on=KEY, validate="one_to_one")

alps_outcome = "ALPS_bilateral_mean_of_ratios"
require(alps, [alps_outcome], "ALPS")
base = base.merge(
    alps[KEY + [alps_outcome, protocol_col]].rename(
        columns={protocol_col: "protocol"}
    ),
    on=KEY, validate="one_to_one"
)

fs_primary = [
    "hippocampus_volume_ml",
    "medial_temporal_composite_thickness_mm",
    "frontal_composite_thickness_mm",
    "temporal_composite_thickness_mm",
    "anterior_cingulate_composite_thickness_mm",
    "insula_thickness_mm",
    "lateral_ventricle_volume_ml",
]
fs_secondary = [
    "cortical_gray_volume_ml",
    "cerebral_white_matter_volume_ml",
    "subcortical_gray_volume_ml",
    "amygdala_volume_ml",
    "corpus_callosum_volume_ml",
    "bilateral_MeanThickness_mm",
    "orbitofrontal_composite_thickness_mm",
]
fs_features = fs_primary + fs_secondary
require(fs, fs_features + ["eTIV_ml"], "FreeSurfer")
base = base.merge(
    fs[KEY + fs_features + ["eTIV_ml"]],
    on=KEY, validate="one_to_one"
)

for col in ["age", "wmh_volume_ml", "eTIV_ml", alps_outcome] + fs_features:
    base[col] = pd.to_numeric(base[col], errors="coerce")
    if not np.isfinite(base[col]).all():
        raise RuntimeError(f"Nonfinite values: {col}")

if (base["wmh_volume_ml"] <= 0).any():
    raise RuntimeError("Unresolved zero/negative WMH entered candidate sample")
if (base["eTIV_ml"] <= 0).any():
    raise RuntimeError("Invalid eTIV")

for col in ["site", "sex"]:
    base[col] = check_categories(base[col], col)
for dataset, col in [("ADNI", "scanner_model"), ("NIFD", "protocol")]:
    idx = base["dataset"].eq(dataset)
    base.loc[idx, col] = check_categories(base.loc[idx, col], col)

base["log1p_WMH_ml"] = np.log1p(base["wmh_volume_ml"])

# Verify all 20 tracts and all four unweighted means.
metrics = ["FA", "MD", "RD", "AD"]
dti_columns = [f"{m}_mean" for m in metrics]
require(dti, dti_columns, "DTI")
dti = base[KEY + ["group"]].merge(
    dti, on=KEY, how="left", suffixes=("_cohort", ""),
    validate="one_to_many"
)
if not dti["group_cohort"].eq(dti["group"]).all():
    raise RuntimeError("DTI group mismatch")

tracts = sorted(dti["tract"].dropna().unique())
if len(tracts) != 20:
    raise RuntimeError(f"Expected 20 tracts; found {len(tracts)}")
if not dti.groupby(KEY).size().eq(20).all():
    raise RuntimeError("A candidate lacks the complete 20-tract set")

eligible = dti["primary_analysis_eligible"].astype(str).str.lower().isin(
    ["true", "1", "1.0", "yes", "pass"]
)
if not eligible.all():
    raise RuntimeError("An ineligible DTI tract entered the common cohort")
for col in dti_columns:
    dti[col] = pd.to_numeric(dti[col], errors="coerce")
    if not np.isfinite(dti[col]).all():
        raise RuntimeError(f"Nonfinite DTI values: {col}")

# Build supported-site subset from the expanded common sample.
required_groups = {
    "NIFD": {"CN", "bvFTD"},
    "ADNI": {"CN", "MCI", "AD"},
}
support = {}
site_records = []
for (dataset, site), part in base.groupby(["dataset", "site"]):
    supported = required_groups[dataset].issubset(set(part["group"]))
    support[(dataset, site)] = supported
    counts = part["group"].value_counts()
    site_records.append({
        "dataset": dataset, "site": site,
        "supported": supported,
        **{f"n_{g}": int(counts.get(g, 0))
           for g in ["CN", "MCI", "AD", "bvFTD"]}
    })

base["supported_site"] = [
    support[(d, s)] for d, s in zip(base["dataset"], base["site"])
]
supported_counts = base.loc[base.supported_site].groupby(
    ["dataset", "group"]
).size().to_dict()
expected_supported = {
    ("ADNI", "CN"): 76, ("ADNI", "MCI"): 83, ("ADNI", "AD"): 39,
    ("NIFD", "CN"): 67, ("NIFD", "bvFTD"): 42,
}
if supported_counts != expected_supported:
    raise RuntimeError(
        f"Supported-site counts differ from audit: {supported_counts}"
    )

save(base, "COMMON_subject_measurements_and_membership")
save(pd.DataFrame(site_records), "COMMON_site_support")

samples = {
    "SUPPORTED_COMMON_307": base.loc[base.supported_site].copy(),
    "EXPANDED_COMMON_337": base.copy(),
}

counts = []
for analysis, frame in samples.items():
    for (dataset, group), n in frame.groupby(["dataset", "group"]).size().items():
        counts.append({
            "analysis": analysis, "dataset": dataset, "group": group, "n": n
        })
save(pd.DataFrame(counts), "COMMON_group_counts")
print(pd.DataFrame(counts).to_string(index=False))

# Outcome specification. eTIV is a covariate, not an outcome.
specs = [
    {"outcome": alps_outcome, "family": "ALPS",
     "modality": "ALPS", "column": alps_outcome,
     "tract": "", "metric": "", "volume": False},
    {"outcome": "log1p_WMH_ml", "family": "WMH",
     "modality": "WMH", "column": "log1p_WMH_ml",
     "tract": "", "metric": "", "volume": False},
]
for feature in fs_features:
    specs.append({
        "outcome": feature,
        "family": "FS_PRIMARY" if feature in fs_primary else "FS_SECONDARY",
        "modality": "FS", "column": feature,
        "tract": "", "metric": "", "volume": feature.endswith("_volume_ml"),
    })
for tract in tracts:
    for metric in metrics:
        specs.append({
            "outcome": f"{tract}::{metric}_mean",
            "family": "DTI", "modality": "DTI",
            "column": f"{metric}_mean", "tract": tract,
            "metric": metric, "volume": False,
        })
save(pd.DataFrame(specs), "COMMON_outcome_definitions")

raw_rows, diagnostic_rows, contrast_rows, adjusted_rows = [], [], [], []

for analysis, sample in samples.items():
    for dataset in ["NIFD", "ADNI"]:
        cohort = sample.loc[sample.dataset.eq(dataset)].copy()
        groups = ["CN", "bvFTD"] if dataset == "NIFD" else ["CN", "MCI", "AD"]
        cohort["group"] = pd.Categorical(cohort["group"], categories=groups)
        cohort["age_c"] = cohort["age"] - cohort["age"].mean()
        cohort["eTIV_c"] = cohort["eTIV_ml"] - cohort["eTIV_ml"].mean()

        for spec in specs:
            frame = cohort.copy()
            if spec["modality"] == "DTI":
                values = dti.loc[dti.tract.eq(spec["tract"]),
                                 KEY + [spec["column"]]].rename(
                    columns={spec["column"]: "y"}
                )
                frame = frame.merge(values, on=KEY, validate="one_to_one")
            else:
                frame["y"] = frame[spec["column"]]

            if len(frame) != len(cohort) or not np.isfinite(frame["y"]).all():
                raise RuntimeError("A model would change the fixed participant set")

            context = {
                "analysis": analysis, "dataset": dataset,
                "family": spec["family"], "outcome": spec["outcome"],
                "tract": spec["tract"], "metric": spec["metric"],
                "n": len(frame),
            }

            for group in groups:
                values = frame.loc[frame.group.eq(group), "y"]
                raw_rows.append({
                    **context, "group": group, "n_group": len(values),
                    "mean": values.mean(), "SD": values.std(ddof=1),
                    "median": values.median(),
                    "Q1": values.quantile(.25), "Q3": values.quantile(.75),
                })
                if spec["modality"] == "WMH":
                    native = frame.loc[frame.group.eq(group), "wmh_volume_ml"]
                    raw_rows.append({
                        **context, "outcome": "WMH_native_ml",
                        "group": group, "n_group": len(native),
                        "mean": native.mean(), "SD": native.std(ddof=1),
                        "median": native.median(),
                        "Q1": native.quantile(.25), "Q3": native.quantile(.75),
                    })

            formula = "y ~ C(group, Treatment(reference='CN')) + age_c + C(sex) + C(site)"
            if spec["modality"] in {"DTI", "ALPS"}:
                formula += " + C(protocol)" if dataset == "NIFD" else " + C(scanner_model)"
            if spec["volume"]:
                formula += " + eTIV_c"

            ordinary = smf.ols(formula, data=frame, missing="raise").fit()
            X = np.asarray(ordinary.model.exog)
            rank = np.linalg.matrix_rank(X)
            p = X.shape[1]
            leverage = np.sum(X * np.linalg.pinv(X).T, axis=1)
            max_h = float(leverage.max())
            valid = rank == p and ordinary.df_resid > 0 and max_h < 1 - 1e-8

            status = (
                "BLOCKED_RANK_DEFICIENT" if rank < p else
                "BLOCKED_NO_RESIDUAL_DF" if ordinary.df_resid <= 0 else
                "BLOCKED_NEAR_UNIT_LEVERAGE" if max_h >= 1 - 1e-8 else
                "PASS_REVIEW_INFLUENCE"
            )
            cov = None
            if valid:
                robust = ordinary.get_robustcov_results(cov_type="HC3", use_t=True)
                cov = np.asarray(robust.cov_params())
                if not np.isfinite(cov).all():
                    valid = False
                    status = "BLOCKED_NONFINITE_HC3"
                    cov = None

            diagnostic_rows.append({
                **context, "parameters": p, "rank": rank,
                "residual_df": ordinary.df_resid,
                "max_leverage": max_h,
                "high_leverage_count": int((leverage > 2*p/len(frame)).sum()),
                "condition_number": ordinary.condition_number,
                "status": status, "formula": formula,
            })

            # Marginal predictions averaged over each dataset/sample's
            # observed covariate distribution, with group assigned in turn.
            mean_design = {}
            for group in groups:
                assigned = frame.copy()
                assigned["group"] = pd.Categorical(
                    [group] * len(frame), categories=groups
                )
                xp = np.asarray(build_design_matrices(
                    [ordinary.model.data.design_info], assigned
                )[0])
                mean_design[group] = xp.mean(axis=0)

                if valid:
                    v = mean_design[group]
                    mean = float(v @ ordinary.params)
                    se = math.sqrt(max(0, float(v @ cov @ v)))
                    crit = stats.t.ppf(.975, ordinary.df_resid)
                    adjusted_rows.append({
                        **context, "group": group,
                        "adjusted_mean_model_scale": mean,
                        "SE_HC3": se,
                        "CI_low": mean - crit*se,
                        "CI_high": mean + crit*se,
                        "status": status,
                    })

            for disease in groups[1:]:
                c = mean_design[disease] - mean_design["CN"]
                # Do not report pseudoinverse estimates as accepted
                # contrasts when model rank/inference is invalid.
                estimate = se = low = high = pvalue = np.nan
                if valid:
                    estimate = float(c @ ordinary.params)
                    se = math.sqrt(max(0, float(c @ cov @ c)))
                    if se > 0:
                        crit = stats.t.ppf(.975, ordinary.df_resid)
                        low, high = estimate-crit*se, estimate+crit*se
                        pvalue = 2*stats.t.sf(abs(estimate/se), ordinary.df_resid)
                contrast_rows.append({
                    **context, "contrast": f"{disease}_minus_CN",
                    "estimate": estimate, "SE_HC3": se,
                    "CI_low": low, "CI_high": high,
                    "p_value": pvalue, "status": status,
                })

        print(f"Finished {analysis}: {dataset}, n={len(cohort)}")

raw = pd.DataFrame(raw_rows)
diagnostics = pd.DataFrame(diagnostic_rows)
contrasts = fdr(pd.DataFrame(contrast_rows),
                ["analysis", "dataset", "family", "contrast"])
adjusted = pd.DataFrame(adjusted_rows)

save(raw, "COMMON_raw_group_values")
save(diagnostics, "COMMON_model_diagnostics")
save(contrasts, "COMMON_within_dataset_contrasts")
save(adjusted, "COMMON_adjusted_group_values")

# Cross-disease contrasts on native model scales.
cross_rows = []
for analysis in samples:
    for spec in specs:
        chosen = contrasts[
            contrasts.analysis.eq(analysis)
            & contrasts.outcome.eq(spec["outcome"])
        ]
        nifd = chosen[
            chosen.dataset.eq("NIFD")
            & chosen.contrast.eq("bvFTD_minus_CN")
        ]
        if len(nifd) != 1:
            raise RuntimeError("Ambiguous NIFD contrast")
        nrow = nifd.iloc[0]

        for disease in ["AD", "MCI"]:
            adni = chosen[
                chosen.dataset.eq("ADNI")
                & chosen.contrast.eq(f"{disease}_minus_CN")
            ]
            if len(adni) != 1:
                raise RuntimeError("Ambiguous ADNI contrast")
            arow = adni.iloc[0]

            values = [nrow.estimate, arow.estimate, nrow.SE_HC3, arow.SE_HC3]
            ok = np.isfinite(values).all()
            diff = se = low = high = pvalue = np.nan
            if ok:
                diff = nrow.estimate - arow.estimate
                se = math.hypot(nrow.SE_HC3, arow.SE_HC3)
                if se > 0:
                    low, high = diff-1.95996398454*se, diff+1.95996398454*se
                    pvalue = 2*stats.norm.sf(abs(diff/se))
            cross_rows.append({
                "analysis": analysis, "family": spec["family"],
                "outcome": spec["outcome"], "tract": spec["tract"],
                "metric": spec["metric"],
                "comparison": f"bvFTD_effect_minus_{disease}_effect",
                "NIFD_effect": nrow.estimate,
                "ADNI_effect": arow.estimate,
                "difference_of_effects": diff, "SE_independent": se,
                "CI_low": low, "CI_high": high, "p_value": pvalue,
                "status": "PASS_REVIEW_INFLUENCE" if ok else "BLOCKED_SOURCE_MODEL",
            })

cross = fdr(pd.DataFrame(cross_rows),
            ["analysis", "family", "comparison"])
save(cross, "COMMON_cross_disease_contrasts")

# Descriptive comparison of overlapping samples: no independent-sample
# test between expanded and supported estimates.
join_cols = ["dataset", "family", "outcome", "contrast"]
fields = join_cols + ["n", "estimate", "CI_low", "CI_high", "BH_q", "status"]
left = contrasts.loc[
    contrasts.analysis.eq("SUPPORTED_COMMON_307"), fields
]
right = contrasts.loc[
    contrasts.analysis.eq("EXPANDED_COMMON_337"), fields
]
comparison = left.merge(
    right, on=join_cols, suffixes=("_supported", "_expanded"),
    validate="one_to_one"
)
comparison["expanded_minus_supported_estimate"] = (
    comparison["estimate_expanded"] - comparison["estimate_supported"]
)
save(comparison, "COMMON_expanded_vs_supported_estimates")

summary = [
    "COMMON02: MULTIMODAL STATISTICAL REFITS",
    "",
    "Expanded common sample: 337; ADNI=225, NIFD=112.",
    "Supported common sample: 307; ADNI=198, NIFD=109.",
    "",
    "Every outcome within a dataset/sample uses the same participants.",
    "DTI extraction: unweighted FA_mean, MD_mean, RD_mean, AD_mean.",
    "Earlier DTI extraction equivalence has NOT been established.",
    "This is a statistical sensitivity run; anatomical QC is not certified.",
    "",
    "MODELS",
    "All: diagnosis + age + sex + site.",
    "DTI/ALPS additionally: NIFD protocol or ADNI scanner.",
    "Structural volumes additionally: eTIV.",
    "WMH outcome: log(1 + WMH volume in mL).",
    "",
    "INFERENCE",
    "Within-dataset: HC3 covariance and residual-df t inference.",
    "Cross-disease: independent coefficient variances, normal approximation.",
    "FDR: within analysis/family/disease contrast or cross-disease comparison.",
    "These FDR families are new-run definitions, not assumed identical to prior scripts.",
    "A blocked model prevents a complete FDR family from being reported.",
    "PASS_REVIEW_INFLUENCE does not certify robustness to influential observations.",
    "No inference is accepted for rank-deficient or near-unit-leverage models.",
    "",
    "DIAGNOSTIC COUNTS",
    diagnostics.groupby(["analysis", "dataset", "status"]).size().to_string(),
    "",
    "NEXT INTERPRETATION",
    "Compare effect sizes and CIs, not significance alone.",
    "Do not choose the preferred sample based on favorable p-values.",
    "Expanded and supported estimates are dependent because subjects overlap.",
    "No preprocessing, source measurements or earlier results were modified.",
]
(OUT / "COMMON02_summary.txt").write_text("\n".join(summary)+"\n")
(OUT / "software_versions.txt").write_text(
    f"Python={platform.python_version()}\n"
    f"NumPy={np.__version__}\npandas={pd.__version__}\n"
    f"SciPy={scipy.__version__}\nstatsmodels={statsmodels.__version__}\n"
)
with (OUT / "input_SHA256SUMS.txt").open("w") as handle:
    for name in ["CANDIDATES", "ALPS", "WMH", "DTI", "FS"]:
        path = Path(os.environ[name])
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")

print("\n" + "\n".join(summary))
print("\n===== ALPS AND WMH WITHIN-DATASET RESULTS =====")
print(contrasts.loc[
    contrasts.family.isin(["ALPS", "WMH"]),
    ["analysis", "dataset", "outcome", "contrast", "n",
     "estimate", "CI_low", "CI_high", "p_value", "BH_q", "status"]
].to_string(index=False))

print("\n===== CROSS-DISEASE TEST COUNTS =====")
for keys, part in cross.groupby(["analysis", "family", "comparison"]):
    tested = part.p_value.notna().sum()
    significant = (part.BH_q < .05).sum()
    print(*keys, f"valid={tested}/{len(part)}",
          f"q<0.05={significant}", sep="\t")

print(f"\nOUTPUT: {OUT}")
