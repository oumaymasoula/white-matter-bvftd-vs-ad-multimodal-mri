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

run = Path(os.environ["RUN"])
out = Path(os.environ["OUT"])
tables = out / "tables"
KEY = ["dataset", "subject"]

def save(frame, name):
    frame.to_csv(tables / f"{name}.tsv", sep="\t", index=False)
    frame.to_csv(tables / f"{name}.csv", index=False)

def yes(series):
    return (
        series.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "pass"})
    )

def unique(frame, columns, label):
    if frame.duplicated(columns).any():
        raise RuntimeError(f"Duplicate keys in {label}: {columns}")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

subjects_path = run / "tables/COMMON_subject_measurements_and_membership.tsv"
reference_path = run / "tables/COMMON_within_dataset_contrasts.tsv"
base = pd.read_csv(subjects_path, sep="\t")
reference = pd.read_csv(reference_path, sep="\t")

for c in KEY + ["group", "sex", "site", "protocol", "scanner_model"]:
    base[c] = base[c].astype(str).str.strip()

unique(base, KEY, "COMMON subjects")
base = base.loc[yes(base["supported_site"])].copy()

expected = {
    ("NIFD", "CN"): 67, ("NIFD", "bvFTD"): 42,
    ("ADNI", "CN"): 76, ("ADNI", "MCI"): 83, ("ADNI", "AD"): 39,
}
if base.groupby(["dataset", "group"]).size().to_dict() != expected:
    raise RuntimeError("Supported participant counts differ from COMMON02")

# Identify DTI input from ORIGINAL provenance, not a filename guess.
dti_candidates = []
for line in (run / "input_SHA256SUMS.txt").read_text().splitlines():
    expected_hash, source = line.split(None, 1)
    path = Path(source.strip())
    if not path.is_file():
        raise RuntimeError(f"Original input no longer exists: {path}")
    if digest(path) != expected_hash:
        raise RuntimeError(f"Original input checksum changed: {path}")
    header = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    if {"dataset", "subject", "group", "tract", "FA_mean",
        "primary_analysis_eligible"}.issubset(header):
        dti_candidates.append(path)

if len(dti_candidates) != 1:
    raise RuntimeError(f"Expected one provenance-verified DTI source: {dti_candidates}")

dti_path = dti_candidates[0]
dti = pd.read_csv(dti_path, sep="\t")
for c in KEY + ["group", "tract"]:
    dti[c] = dti[c].astype(str).str.strip()
dti["group"] = dti["group"].replace({"Probable_AD": "AD"})
unique(dti, KEY + ["tract"], "DTI")

dti = base[KEY + ["group"]].merge(
    dti, on=KEY, how="left", suffixes=("_cohort", ""),
    validate="one_to_many"
)
if not dti["group_cohort"].eq(dti["group"]).all():
    raise RuntimeError("DTI diagnostic labels disagree with COMMON02")
if not dti.groupby(KEY).size().eq(20).all():
    raise RuntimeError("Incomplete 20-tract input")
if not yes(dti["primary_analysis_eligible"]).all():
    raise RuntimeError("Ineligible DTI rows in supported cohort")

selected = {
    "FA_forceps_minor": ["Forceps minor"],
    "FA_uncinate_bilateral": [
        "Uncinate fasciculus L", "Uncinate fasciculus R"
    ],
    "FA_ATR_bilateral": [
        "Anterior thalamic radiation L",
        "Anterior thalamic radiation R"
    ],
}

# Merge one tract at a time, preserving the original unweighted means.
for output_name, tracts in selected.items():
    parts = []
    for i, tract in enumerate(tracts):
        part = dti.loc[dti["tract"].eq(tract), KEY + ["FA_mean"]].copy()
        if len(part) != len(base):
            raise RuntimeError(f"Missing or ambiguous tract: {tract}")
        unique(part, KEY, tract)
        part["FA_mean"] = pd.to_numeric(part["FA_mean"], errors="raise")
        col = f"{output_name}_{i}"
        base = base.merge(
            part.rename(columns={"FA_mean": col}),
            on=KEY, validate="one_to_one"
        )
        parts.append(col)
    base[output_name] = base[parts].mean(axis=1)

primary = [
    "hippocampus_volume_ml",
    "medial_temporal_composite_thickness_mm",
    "frontal_composite_thickness_mm",
    "temporal_composite_thickness_mm",
    "anterior_cingulate_composite_thickness_mm",
    "insula_thickness_mm",
    "lateral_ventricle_volume_ml",
]
outcomes = primary + [
    "FA_forceps_minor", "FA_uncinate_bilateral", "FA_ATR_bilateral",
    "log1p_WMH_ml", "ALPS_bilateral_mean_of_ratios",
]
numeric = outcomes + ["age", "eTIV_ml",
                     "orbitofrontal_composite_thickness_mm"]
for c in numeric:
    base[c] = pd.to_numeric(base[c], errors="raise")
    if not np.isfinite(base[c]).all():
        raise RuntimeError(f"Nonfinite input: {c}")

save(base, "targeted_subject_measurements")

diagnostics = []
influence = []
rows = []
counts = []

def fit_checked(frame, formula, context, record_influence=False):
    ordinary = smf.ols(formula, data=frame, missing="raise").fit()
    X = np.asarray(ordinary.model.exog)
    rank = np.linalg.matrix_rank(X)
    p = X.shape[1]
    h = np.sum(X * np.linalg.pinv(X).T, axis=1)

    status = (
        "BLOCKED_RANK_DEFICIENT" if rank < p else
        "BLOCKED_NO_RESIDUAL_DF" if ordinary.df_resid <= 0 else
        "BLOCKED_NEAR_UNIT_LEVERAGE" if h.max() >= 1 - 1e-8 else
        "PASS_REVIEW_INFLUENCE"
    )
    robust = None
    if status == "PASS_REVIEW_INFLUENCE":
        robust = ordinary.get_robustcov_results(cov_type="HC3", use_t=True)
        if not np.isfinite(robust.cov_params()).all():
            status = "BLOCKED_NONFINITE_HC3"
            robust = None

    diagnostics.append({
        **context, "n": len(frame), "formula": formula,
        "parameters": p, "rank": rank,
        "residual_df": ordinary.df_resid,
        "max_leverage": float(h.max()),
        "condition_number": float(ordinary.condition_number),
        "status": status,
    })

    if record_influence and robust is not None:
        cook = ordinary.get_influence().cooks_distance[0]
        for subject, hi, ci in zip(frame["subject"], h, cook):
            influence.append({
                **context, "subject": subject,
                "leverage": hi, "Cooks_D": ci,
                "leverage_flag_2p_over_n": bool(hi > 2*p/len(frame)),
                "Cooks_flag_4_over_n": bool(ci > 4/len(frame)),
            })
    return ordinary, robust, status

def coefficient(ordinary, robust, term):
    idx = ordinary.model.exog_names.index(term)
    beta = float(robust.params[idx])
    se = float(robust.bse[idx])
    if not np.isfinite(se) or se <= 0:
        return [np.nan] * 5
    critical = stats.t.ppf(.975, ordinary.df_resid)
    return [
        beta, se, beta-critical*se, beta+critical*se,
        2*stats.t.sf(abs(beta/se), ordinary.df_resid)
    ]

groups_by_dataset = {
    "NIFD": ["CN", "bvFTD"], "ADNI": ["CN", "MCI", "AD"]
}

def run_variant(name, frame, omit_acquisition=False):
    for dataset, groups in groups_by_dataset.items():
        cohort = frame.loc[frame["dataset"].eq(dataset)].copy()
        for group in groups:
            counts.append({
                "variant": name, "dataset": dataset, "group": group,
                "n": int(cohort["group"].eq(group).sum())
            })
        group_counts = cohort["group"].value_counts()
        enough = all(group_counts.get(g, 0) >= 5 for g in groups)
        cohort["group"] = pd.Categorical(cohort["group"], categories=groups)
        cohort["age_c"] = cohort["age"] - cohort["age"].mean()
        cohort["eTIV_c"] = cohort["eTIV_ml"] - cohort["eTIV_ml"].mean()

        for outcome in outcomes:
            context = {
                "variant": name, "dataset": dataset, "outcome": outcome
            }
            ordinary = robust = None
            status = "BLOCKED_GROUP_N_LT_5"
            if enough:
                cohort["y"] = cohort[outcome]
                formula = (
                    "y ~ C(group, Treatment(reference='CN'))"
                    " + age_c + C(sex) + C(site)"
                )
                is_diffusion = (
                    outcome.startswith("FA_") or outcome.startswith("ALPS_")
                )
                if is_diffusion and not omit_acquisition:
                    formula += (
                        " + C(protocol)" if dataset == "NIFD"
                        else " + C(scanner_model)"
                    )
                if outcome.endswith("_volume_ml"):
                    formula += " + eTIV_c"
                ordinary, robust, status = fit_checked(
                    cohort, formula, context,
                    record_influence=(name == "BASELINE")
                )
            for disease in groups[1:]:
                values = [np.nan] * 5
                if robust is not None:
                    term = f"C(group, Treatment(reference='CN'))[T.{disease}]"
                    values = coefficient(ordinary, robust, term)
                rows.append({
                    **context, "contrast": f"{disease}_minus_CN",
                    "n": len(cohort),
                    "n_CN": int(group_counts.get("CN", 0)),
                    "n_disease": int(group_counts.get(disease, 0)),
                    **dict(zip(
                        ["estimate", "SE_HC3", "CI_low", "CI_high", "p_value"],
                        values
                    )),
                    "status": status,
                })

run_variant("BASELINE", base)

# Shared min-max age support across all five groups.
ranges = base.groupby(["dataset", "group"])["age"].agg(["min", "max"])
lower = float(ranges["min"].max())
upper = float(ranges["max"].min())
if lower >= upper:
    raise RuntimeError("No common age interval across five groups")
age_frame = base.loc[base["age"].between(lower, upper)].copy()
run_variant("AGE_OVERLAP", age_frame)
save(age_frame[KEY + ["group", "age", "site"]], "age_overlap_membership")

# Leave each site out; other dataset is retained as its reference.
for dataset in ["NIFD", "ADNI"]:
    for index, site in enumerate(sorted(
        base.loc[base["dataset"].eq(dataset), "site"].unique()
    ), start=1):
        keep = ~(base["dataset"].eq(dataset) & base["site"].eq(site))
        name = f"LOSO_{dataset}_{index:02d}"
        run_variant(name, base.loc[keep].copy())

# NIFD protocol categories containing both CN and bvFTD.
nifd = base.loc[base["dataset"].eq("NIFD")]
protocol_counts = pd.crosstab(nifd["protocol"], nifd["group"])
shared = protocol_counts.index[
    (protocol_counts.get("CN", 0) > 0)
    & (protocol_counts.get("bvFTD", 0) > 0)
].tolist()
keep = ~base["dataset"].eq("NIFD") | base["protocol"].isin(shared)
run_variant("NIFD_SHARED_PROTOCOLS", base.loc[keep].copy())

# Largest shared protocol, selected by sample count, not imaging results.
if shared:
    largest = protocol_counts.loc[shared].sum(axis=1).idxmax()
    keep = ~base["dataset"].eq("NIFD") | base["protocol"].eq(largest)
    run_variant("NIFD_LARGEST_SHARED_PROTOCOL", base.loc[keep].copy())
else:
    largest = "NONE"

# ADNI analyses within each scanner model.
for index, scanner in enumerate(sorted(
    base.loc[base["dataset"].eq("ADNI"), "scanner_model"].unique()
), start=1):
    keep = ~base["dataset"].eq("ADNI") | base["scanner_model"].eq(scanner)
    run_variant(f"ADNI_SCANNER_{index:02d}", base.loc[keep].copy())

within = pd.DataFrame(rows)

# Reproduce existing source results BEFORE interpreting sensitivity models.
checks = []
mapping = {
    **{c: c for c in primary},
    "ALPS_bilateral_mean_of_ratios": "ALPS_bilateral_mean_of_ratios",
    "log1p_WMH_ml": "log1p_WMH_ml",
    "FA_forceps_minor": "Forceps minor::FA_mean",
}
for _, row in within.loc[within["variant"].eq("BASELINE")].iterrows():
    if row["outcome"] not in mapping:
        continue
    original = reference.loc[
        reference["analysis"].eq("SUPPORTED_COMMON_307")
        & reference["dataset"].eq(row["dataset"])
        & reference["contrast"].eq(row["contrast"])
        & reference["outcome"].eq(mapping[row["outcome"]])
    ]
    if len(original) != 1:
        raise RuntimeError("Ambiguous original baseline result")
    source = original.iloc[0]
    fields = ["estimate", "SE_HC3", "CI_low", "CI_high", "p_value"]
    passed = all(np.isclose(
        row[c], source[c], rtol=1e-6, atol=1e-9
    ) for c in fields)
    checks.append({
        "dataset": row["dataset"], "outcome": row["outcome"],
        "contrast": row["contrast"], "reproduced": passed
    })
save(pd.DataFrame(checks), "baseline_reproduction")
if not all(c["reproduced"] for c in checks):
    raise RuntimeError("Baseline reproduction failed; stop interpretation")

def apply_fdr(frame, grouping):
    frame = frame.copy()
    frame["sensitivity_BH_q"] = np.nan
    frame["FDR_family_complete"] = False
    for _, idx in frame.groupby(grouping, dropna=False).groups.items():
        p = frame.loc[idx, "p_value"].to_numpy(dtype=float)
        if np.isfinite(p).all():
            frame.loc[idx, "sensitivity_BH_q"] = multipletests(
                p, method="fdr_bh"
            )[1]
            frame.loc[idx, "FDR_family_complete"] = True
    return frame

within = apply_fdr(within, ["variant", "dataset", "contrast"])
save(within, "within_dataset_sensitivity")

cross_rows = []
for (variant, outcome), part in within.groupby(["variant", "outcome"]):
    nrow = part.loc[
        part["dataset"].eq("NIFD") & part["contrast"].eq("bvFTD_minus_CN")
    ].iloc[0]
    for disease in ["AD", "MCI"]:
        arow = part.loc[
            part["dataset"].eq("ADNI")
            & part["contrast"].eq(f"{disease}_minus_CN")
        ].iloc[0]
        values = [np.nan] * 5
        valid = np.isfinite([
            nrow["estimate"], nrow["SE_HC3"],
            arow["estimate"], arow["SE_HC3"]
        ]).all()
        if valid:
            difference = nrow["estimate"] - arow["estimate"]
            se = math.hypot(nrow["SE_HC3"], arow["SE_HC3"])
            if se > 0:
                z = stats.norm.ppf(.975)
                values = [
                    difference, se, difference-z*se, difference+z*se,
                    2*stats.norm.sf(abs(difference/se))
                ]
        cross_rows.append({
            "variant": variant, "outcome": outcome,
            "comparison": f"bvFTD_effect_minus_{disease}_effect",
            "NIFD_effect": nrow["estimate"],
            "ADNI_effect": arow["estimate"],
            **dict(zip(
                ["estimate", "SE_independent", "CI_low", "CI_high", "p_value"],
                values
            )),
            "status": "PASS_REVIEW_INFLUENCE" if valid
                      else "BLOCKED_SOURCE_MODEL",
        })
cross = apply_fdr(
    pd.DataFrame(cross_rows), ["variant", "comparison"]
)
save(cross, "cross_disease_sensitivity")

# Descriptive change from baseline: overlapping samples are NOT independent.
baseline = cross.loc[cross["variant"].eq("BASELINE"), [
    "outcome", "comparison", "estimate", "CI_low", "CI_high"
]].rename(columns={
    "estimate": "baseline_estimate",
    "CI_low": "baseline_CI_low", "CI_high": "baseline_CI_high"
})
comparison = cross.merge(
    baseline, on=["outcome", "comparison"], validate="many_to_one"
)
comparison["change_from_baseline"] = (
    comparison["estimate"] - comparison["baseline_estimate"]
)
comparison["same_direction_as_baseline"] = np.where(
    comparison["estimate"].notna(),
    np.sign(comparison["estimate"]) == np.sign(comparison["baseline_estimate"]),
    pd.NA
)
save(comparison, "sensitivity_vs_baseline")

# Six exploratory within-patient association tests:
# two anatomically motivated pairs in bvFTD, AD and MCI.
pairs = [
    ("frontal_forceps",
     "frontal_composite_thickness_mm", "FA_forceps_minor"),
    ("orbitofrontal_uncinate",
     "orbitofrontal_composite_thickness_mm", "FA_uncinate_bilateral"),
]
association_rows = []
for dataset, group in [("NIFD", "bvFTD"), ("ADNI", "AD"), ("ADNI", "MCI")]:
    frame = base.loc[
        base["dataset"].eq(dataset) & base["group"].eq(group)
    ].copy()
    frame["age_c"] = frame["age"] - frame["age"].mean()
    for pair, structural, diffusion in pairs:
        context = {
            "variant": "WITHIN_PATIENT_ASSOCIATION",
            "dataset": dataset, "group": group, "outcome": pair
        }
        frame["y"] = frame[diffusion]
        sd = frame[structural].std(ddof=1)
        values = [np.nan] * 5
        status = "BLOCKED_CONSTANT_PREDICTOR"
        if np.isfinite(sd) and sd > 0:
            frame["x"] = (frame[structural] - frame[structural].mean()) / sd
            formula = "y ~ x + age_c + C(sex) + C(site)"
            formula += (
                " + C(protocol)" if dataset == "NIFD"
                else " + C(scanner_model)"
            )
            ordinary, robust, status = fit_checked(
                frame, formula, context, record_influence=True
            )
            if robust is not None:
                values = coefficient(ordinary, robust, "x")
        association_rows.append({
            **context, "n": len(frame),
            "structural_predictor": structural,
            "diffusion_outcome": diffusion,
            "interpretation": "FA change per 1 within-group SD higher thickness",
            **dict(zip(
                ["estimate", "SE_HC3", "CI_low", "CI_high", "p_value"],
                values
            )), "status": status
        })

associations = pd.DataFrame(association_rows)
associations["family"] = "SIX_EXPLORATORY_ASSOCIATIONS"
associations = apply_fdr(associations, ["family"])
save(associations, "within_patient_crossmodal_associations")
save(pd.DataFrame(diagnostics), "model_diagnostics")
save(pd.DataFrame(influence), "influence_audit")
save(pd.DataFrame(counts), "variant_group_counts")

# Variant definitions, including names that would otherwise be opaque.
definitions = [
    {"variant": "BASELINE", "restriction": "Original supported 307"},
    {"variant": "AGE_OVERLAP",
     "restriction": f"All groups: {lower:.6f} <= age <= {upper:.6f}"},
    {"variant": "NIFD_SHARED_PROTOCOLS",
     "restriction": "NIFD protocols with both diagnoses: " + "; ".join(shared)},
    {"variant": "NIFD_LARGEST_SHARED_PROTOCOL",
     "restriction": str(largest)},
]
for dataset in ["NIFD", "ADNI"]:
    for i, site in enumerate(sorted(
        base.loc[base["dataset"].eq(dataset), "site"].unique()
    ), start=1):
        definitions.append({
            "variant": f"LOSO_{dataset}_{i:02d}",
            "restriction": f"Exclude {dataset} site: {site}"
        })
for i, scanner in enumerate(sorted(
    base.loc[base["dataset"].eq("ADNI"), "scanner_model"].unique()
), start=1):
    definitions.append({
        "variant": f"ADNI_SCANNER_{i:02d}",
        "restriction": f"ADNI scanner only: {scanner}"
    })
save(pd.DataFrame(definitions), "variant_definitions")

notes = f"""
Exploratory targeted sensitivity analyses.
Original input checksums verified.
Original HC3 disease-control results reproduced before sensitivity reporting.
DTI uses original unweighted FA_mean.
Bilateral FA = arithmetic mean of left and right tract means.
Age overlap: {lower:.6f} to {upper:.6f} years; not age matching.
Original supported-site membership retained before sensitivity restrictions.
Sites are not reselected after age restriction.
Models blocked for rank deficiency, near-unit leverage, or fewer than
five subjects in any required group. Five is a blocking rule, not a
claim of adequate power.
Within-model inference: HC3, residual-df t.
Cross-dataset inference: independent coefficient variances, normal approximation.
Sensitivity FDR: 12 outcomes per variant/disease-control or cross-disease comparison.
Association FDR: all six within-patient tests; incomplete families withheld.
These FDR definitions do not replace original COMMON02 FDR results.
Single-protocol/scanner covariates become constant within restricted samples.
Influence flags are descriptive; no participants removed using imaging results.
Within-patient associations are exploratory, not causal or independent validation.
No clinical severity sensitivity or regional subcortical refitting performed.
No direct pooled bvFTD-versus-AD classifier fitted.
Age, site and acquisition sensitivities do not eliminate dataset confounding.
Do not select a preferred variant using p-values.
No source measurements or previous results modified.
"""
(out / "README.txt").write_text(notes.strip() + "\n")
(out / "software_versions.txt").write_text(
    f"Python={platform.python_version()}\nNumPy={np.__version__}\n"
    f"pandas={pd.__version__}\nSciPy={scipy.__version__}\n"
    f"statsmodels={statsmodels.__version__}\n"
)
(out / "input_checksums.txt").write_text(
    "\n".join(f"{digest(p)}  {p}" for p in [
        subjects_path, reference_path, dti_path
    ]) + "\n"
)

print("===== BASELINE REPRODUCTION =====")
print(pd.DataFrame(checks).to_string(index=False))
print("\n===== AGE OVERLAP COUNTS =====")
print(pd.DataFrame(counts).query("variant == 'AGE_OVERLAP'").to_string(index=False))
print("\n===== MODEL STATUS =====")
print(pd.DataFrame(diagnostics).groupby(
    ["variant", "dataset", "status"]
).size().to_string())
print("\n===== FRONTAL CROSS-DISEASE SENSITIVITY =====")
print(cross.loc[
    cross["outcome"].eq("frontal_composite_thickness_mm")
].to_string(index=False))
print("\n===== WITHIN-PATIENT ASSOCIATIONS =====")
print(associations.to_string(index=False))
print("\nOUTPUT:", out)
