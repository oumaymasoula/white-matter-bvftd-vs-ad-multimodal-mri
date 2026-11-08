import os
import io
import re
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

out = Path(os.environ["OUT"])
KEY = ["dataset", "subject"]

def save(frame, name):
    frame.to_csv(out / "tables" / f"{name}.tsv",
                 sep="\t", index=False)
    frame.to_csv(out / "tables" / f"{name}.csv", index=False)

def yes(series):
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes", "pass"}
    )

def unique(frame, keys, label):
    if frame.duplicated(keys).any():
        raise RuntimeError(f"Duplicate keys: {label}")

def read_original(filename):
    text = Path(os.environ["REVIEW"]).read_text(encoding="utf-8")
    matches = re.findall(
        r"===== ORIGINAL RESULTS: ([^\n]+) =====\n"
        r"(.*?)(?=\n=====|\Z)",
        text, flags=re.S
    )
    bodies = [body for path, body in matches
              if Path(path.strip()).name == filename]
    if len(bodies) != 1:
        raise RuntimeError(f"Expected one original table: {filename}")
    return pd.read_csv(io.StringIO(bodies[0].strip()), sep="\t")

cort_ref = read_original("FreeSurfer_DK68_cortical_thickness_results.tsv")
sub_ref = read_original("FreeSurfer_subcortical_16_results.tsv")

if len(cort_ref) != 204 or len(sub_ref) != 48:
    raise RuntimeError("Unexpected original result counts")

subjects = pd.read_csv(os.environ["SUBJECTS"], sep="\t")
subjects = subjects.loc[yes(subjects["supported_site"])].copy()
for c in KEY + ["group", "sex", "site"]:
    subjects[c] = subjects[c].astype(str).str.strip()
unique(subjects, KEY, "subjects")

expected = {
    ("NIFD", "CN"): 67, ("NIFD", "bvFTD"): 42,
    ("ADNI", "CN"): 76, ("ADNI", "MCI"): 83, ("ADNI", "AD"): 39,
}
if subjects.groupby(["dataset", "group"]).size().to_dict() != expected:
    raise RuntimeError("Unexpected supported cohort")

for c in ["age", "eTIV_ml"]:
    subjects[c] = pd.to_numeric(subjects[c], errors="raise")
    if not np.isfinite(subjects[c]).all():
        raise RuntimeError(f"Nonfinite {c}")

definitions = []
regional_parts = []

for family, ref, source_name, value_col in [
    ("CORTICAL", cort_ref, "APARC", "thickness_mean_mm"),
    ("SUBCORTICAL", sub_ref, "ASEG", "volume_mm3"),
]:
    regional = pd.read_csv(os.environ[source_name], sep="\t")
    for c in KEY + ["group"]:
        regional[c] = regional[c].astype(str).str.strip()
    regional["group"] = regional["group"].replace({"Probable_AD": "AD"})

    if family == "CORTICAL":
        regional["feature"] = (
            regional["hemisphere"].astype(str)
            + "_" + regional["region"].astype(str)
        )
        defs = ref[["hemisphere", "region", "parcel"]].drop_duplicates()
        defs = defs.rename(columns={"parcel": "feature"})
        if len(defs) != 68:
            raise RuntimeError("Expected 68 cortical parcels")
    else:
        regional["feature"] = regional["structure"]
        defs = ref[["hemisphere", "region", "structure"]].drop_duplicates()
        defs = defs.rename(columns={"structure": "feature"})
        if len(defs) != 16:
            raise RuntimeError("Expected 16 subcortical structures")

    defs["family"] = family
    definitions.extend(defs.to_dict("records"))
    regional = regional.loc[regional["feature"].isin(defs["feature"])].copy()
    unique(regional, KEY + ["feature"], family)

    merged = subjects.merge(
        regional[KEY + ["group", "feature", value_col]],
        on=KEY, how="inner", suffixes=("", "_regional"),
        validate="one_to_many"
    )
    if not merged["group"].eq(merged["group_regional"]).all():
        raise RuntimeError(f"Diagnostic label mismatch: {family}")

    merged["y"] = pd.to_numeric(merged[value_col], errors="raise")
    if family == "SUBCORTICAL":
        merged["y"] /= 1000.0
    if not np.isfinite(merged["y"]).all():
        raise RuntimeError(f"Nonfinite regional values: {family}")

    expected_features = 68 if family == "CORTICAL" else 16
    counts = merged.groupby(KEY)["feature"].nunique()
    if len(counts) != 307 or not counts.eq(expected_features).all():
        raise RuntimeError(f"Incomplete regional participant set: {family}")

    merged["family"] = family
    regional_parts.append(merged)

regional = pd.concat(regional_parts, ignore_index=True)
save(pd.DataFrame(definitions), "regional_definitions")

diagnostics, counts, definitions_variants = [], [], []
all_within, all_cross = [], []

def fdr(frame, grouping):
    frame = frame.copy()
    frame["BH_q"] = np.nan
    frame["FDR_family_complete"] = False
    for _, idx in frame.groupby(grouping).groups.items():
        p = frame.loc[idx, "p_value"].to_numpy()
        if np.isfinite(p).all():
            frame.loc[idx, "BH_q"] = multipletests(p, method="fdr_bh")[1]
            frame.loc[idx, "FDR_family_complete"] = True
    return frame

def run_variant(name, sample):
    keys = sample[KEY]
    data = regional.merge(keys, on=KEY, validate="many_to_one")
    rows = []
    for dataset, groups in [
        ("NIFD", ["CN", "bvFTD"]),
        ("ADNI", ["CN", "MCI", "AD"]),
    ]:
        cohort = sample.loc[sample["dataset"].eq(dataset)]
        gc = cohort["group"].value_counts()
        for g in groups:
            counts.append({
                "variant": name, "dataset": dataset,
                "group": g, "n": int(gc.get(g, 0))
            })

        for definition in definitions:
            family, feature = definition["family"], definition["feature"]
            frame = data.loc[
                data["dataset"].eq(dataset)
                & data["family"].eq(family)
                & data["feature"].eq(feature)
            ].copy()
            if len(frame) != len(cohort):
                raise RuntimeError("Model changed participant set")

            status = "BLOCKED_GROUP_N_LT_5"
            robust = None
            sigma = np.nan
            if all(gc.get(g, 0) >= 5 for g in groups):
                formula = (
                    'y ~ C(group, Treatment(reference="CN"))'
                    ' + age + C(sex) + C(site)'
                )
                if family == "SUBCORTICAL":
                    formula += " + eTIV_ml"
                model = smf.ols(formula, data=frame, missing="raise")
                ordinary = model.fit()
                X = np.asarray(ordinary.model.exog)
                rank = np.linalg.matrix_rank(X)
                p = X.shape[1]
                h = ordinary.get_influence().hat_matrix_diag
                sigma = float(np.sqrt(ordinary.mse_resid))
                status = (
                    "BLOCKED_RANK_DEFICIENT" if rank < p else
                    "BLOCKED_NO_RESIDUAL_DF"
                    if ordinary.df_resid <= 0 else
                    "BLOCKED_NEAR_UNIT_LEVERAGE"
                    if h.max() >= .99 else "PASS"
                )
                if status == "PASS":
                    robust = model.fit(cov_type="HC3", use_t=True)
                    if not np.isfinite(np.asarray(robust.cov_params())).all():
                        robust = None
                        status = "BLOCKED_NONFINITE_HC3"
                diagnostics.append({
                    "variant": name, "dataset": dataset,
                    "family": family, "feature": feature,
                    "n": len(frame), "rank": rank, "parameters": p,
                    "residual_df": ordinary.df_resid,
                    "max_leverage": float(h.max()),
                    "residual_SD": sigma,
                    "formula": formula, "status": status,
                })
            else:
                diagnostics.append({
                    "variant": name, "dataset": dataset,
                    "family": family, "feature": feature,
                    "n": len(frame), "status": status
                })

            for disease in groups[1:]:
                b = se = low = high = pv = np.nan
                if robust is not None:
                    term = (
                        'C(group, Treatment(reference="CN"))'
                        f'[T.{disease}]'
                    )
                    b = float(robust.params[term])
                    se = float(robust.bse[term])
                    low, high = robust.conf_int().loc[term].to_numpy()
                    pv = float(robust.pvalues[term])
                rows.append({
                    "variant": name, "dataset": dataset,
                    **definition, "disease": disease,
                    "n": len(frame), "estimate": b, "SE_HC3": se,
                    "CI_low": low, "CI_high": high, "p_value": pv,
                    "standardized_beta_residual_SD":
                        b/sigma if np.isfinite(sigma) and sigma > 0
                        else np.nan,
                    "status": status,
                })

    within = fdr(pd.DataFrame(rows), ["dataset", "family", "disease"])
    cross_rows = []
    for definition in definitions:
        chosen = within.loc[
            within["family"].eq(definition["family"])
            & within["feature"].eq(definition["feature"])
        ]
        nr = chosen.loc[chosen["disease"].eq("bvFTD")].iloc[0]
        for disease in ["AD", "MCI"]:
            ar = chosen.loc[chosen["disease"].eq(disease)].iloc[0]
            b = se = low = high = pv = np.nan
            valid = np.isfinite([
                nr["estimate"], nr["SE_HC3"],
                ar["estimate"], ar["SE_HC3"]
            ]).all()
            if valid:
                b = nr["estimate"] - ar["estimate"]
                se = math.hypot(nr["SE_HC3"], ar["SE_HC3"])
                z = stats.norm.ppf(.975)
                low, high = b-z*se, b+z*se
                pv = 2*stats.norm.sf(abs(b/se))
            cross_rows.append({
                "variant": name, **definition,
                "comparison": f"bvFTD_effect_minus_{disease}_effect",
                "estimate": b, "SE_independent": se,
                "CI_low": low, "CI_high": high, "p_value": pv,
                "status": "PASS" if valid else "BLOCKED_SOURCE_MODEL",
            })
    cross = fdr(pd.DataFrame(cross_rows), ["family", "comparison"])
    return within, cross

baseline_w, baseline_c = run_variant("BASELINE", subjects)

# Validate reconstructed models against all original results and q-values.
checks = []
for family, ref, effect_col, se_col, low_col, high_col in [
    ("CORTICAL", cort_ref, "adjusted_thickness_difference_mm",
     "SE_HC3", "CI_low", "CI_high"),
    ("SUBCORTICAL", sub_ref, "adjusted_volume_difference_ml",
     "SE_HC3_ml", "CI_low_ml", "CI_high_ml"),
]:
    for _, original in ref.iterrows():
        feature = original["parcel"] if family == "CORTICAL" else original["structure"]
        comparison = original["comparison"]
        disease = (
            "bvFTD" if comparison.startswith("bvFTD") else
            "MCI" if comparison.startswith("MCI") else "AD"
        )
        match = baseline_w.loc[
            baseline_w["family"].eq(family)
            & baseline_w["dataset"].eq(original["dataset"])
            & baseline_w["feature"].eq(feature)
            & baseline_w["disease"].eq(disease)
        ]
        if len(match) != 1:
            raise RuntimeError("Ambiguous baseline match")
        fitted = match.iloc[0]
        mapping = {
            "estimate": effect_col, "SE_HC3": se_col,
            "CI_low": low_col, "CI_high": high_col,
            "p_value": "p_HC3", "BH_q": "BH_q",
        }
        for new_col, old_col in mapping.items():
            passed = np.isclose(
                fitted[new_col], original[old_col],
                rtol=1e-5, atol=1e-10
            )
            checks.append({
                "family": family, "feature": feature,
                "disease": disease, "field": new_col,
                "reproduced": bool(passed),
                "new_value": fitted[new_col],
                "original_value": original[old_col],
            })

save(pd.DataFrame(checks), "baseline_reproduction")
if not all(r["reproduced"] for r in checks):
    raise RuntimeError(
        "Baseline reproduction failed. See baseline_reproduction.tsv. "
        "No sensitivity models were run."
    )

all_within.append(baseline_w)
all_cross.append(baseline_c)
definitions_variants.append({
    "variant": "BASELINE", "definition": "Original supported 307"
})

ranges = subjects.groupby(["dataset", "group"])["age"].agg(["min", "max"])
lower, upper = float(ranges["min"].max()), float(ranges["max"].min())
variants = {
    "AGE_OVERLAP": subjects.loc[
        subjects["age"].between(lower, upper)
    ].copy()
}
definitions_variants.append({
    "variant": "AGE_OVERLAP",
    "definition": f"Age {lower:.6f} to {upper:.6f} years"
})

for dataset in ["NIFD", "ADNI"]:
    sites = sorted(subjects.loc[
        subjects["dataset"].eq(dataset), "site"
    ].unique())
    for i, site in enumerate(sites, start=1):
        name = f"LOSO_{dataset}_{i:02d}"
        keep = ~(
            subjects["dataset"].eq(dataset)
            & subjects["site"].eq(site)
        )
        variants[name] = subjects.loc[keep].copy()
        definitions_variants.append({
            "variant": name,
            "definition": f"Exclude {dataset} site: {site}"
        })

for name, sample in variants.items():
    w, c = run_variant(name, sample)
    all_within.append(w)
    all_cross.append(c)
    print("Finished:", name, flush=True)

within = pd.concat(all_within, ignore_index=True)
cross = pd.concat(all_cross, ignore_index=True)
save(within, "regional_within_dataset")
save(cross, "regional_cross_disease")
save(pd.DataFrame(diagnostics), "regional_model_diagnostics")
save(pd.DataFrame(counts), "variant_group_counts")
save(pd.DataFrame(definitions_variants), "variant_definitions")

# Exploratory original enrichment criterion, evaluated in complete families.
selection_rows = []
for variant in within["variant"].unique():
    for definition in definitions:
        family, feature = definition["family"], definition["feature"]
        w = within.loc[
            within["variant"].eq(variant)
            & within["family"].eq(family)
            & within["feature"].eq(feature)
            & within["disease"].eq("bvFTD")
        ].iloc[0]
        cs = cross.loc[
            cross["variant"].eq(variant)
            & cross["family"].eq(family)
            & cross["feature"].eq(feature)
        ]
        ad = cs.loc[
            cs["comparison"].eq("bvFTD_effect_minus_AD_effect")
        ].iloc[0]
        mci = cs.loc[
            cs["comparison"].eq("bvFTD_effect_minus_MCI_effect")
        ].iloc[0]

        # Smaller tissue volume/thickness; larger lateral ventricles.
        direction = 1 if "Lateral-Ventricle" in feature else -1
        available = np.isfinite([w["BH_q"], ad["BH_q"], mci["BH_q"]]).all()
        selected = (
            available
            and w["BH_q"] < .05
            and ad["BH_q"] < .05
            and mci["BH_q"] < .05
            and direction*w["estimate"] > 0
            and direction*ad["estimate"] > 0
            and direction*mci["estimate"] > 0
        )
        selection_rows.append({
            "variant": variant, **definition,
            "bvFTD_effect": w["estimate"],
            "bvFTD_q": w["BH_q"],
            "bvFTD_minus_AD": ad["estimate"],
            "bvFTD_minus_AD_CI_low": ad["CI_low"],
            "bvFTD_minus_AD_CI_high": ad["CI_high"],
            "bvFTD_minus_AD_q": ad["BH_q"],
            "bvFTD_minus_MCI": mci["estimate"],
            "bvFTD_minus_MCI_q": mci["BH_q"],
            "criterion_evaluable": available,
            "strict_enriched": bool(selected),
        })

selection = pd.DataFrame(selection_rows)
save(selection, "regional_enrichment_by_variant")
baseline_selected = selection.loc[
    selection["variant"].eq("BASELINE") & selection["strict_enriched"],
    ["family", "feature"]
]
tracked = selection.merge(
    baseline_selected, on=["family", "feature"], validate="many_to_one"
)
save(tracked, "original_enriched_regions_sensitivity")

(out / "README.txt").write_text(
    "All 204 cortical and 48 subcortical original results reproduced "
    "before sensitivity refitting; coefficient, SE, CI, p and q checked.\n"
    "Cortical model reconstructed: diagnosis + age + sex + site.\n"
    "Subcortical model: diagnosis + age + sex + site + eTIV.\n"
    "HC3 and residual-df t inference within datasets.\n"
    "Cross-disease effects: independent coefficient variances, normal inference.\n"
    "FDR across all 68 cortical or 16 subcortical tests per contrast "
    "and variant; incomplete families withheld.\n"
    "Exploratory enrichment requires all three corrected contrasts "
    "in the expected direction. This does not establish composite-pattern FDR.\n"
    "Sensitivity variants: age overlap and leave-one-site-out.\n"
    "No T1 scanner adjustment added without T1-specific metadata.\n"
    "Models blocked at rank deficiency, leverage >=0.99, "
    "or required group size <5. This cutoff does not establish adequate power.\n"
    "Age overlap is not age matching; clinical severity not adjusted.\n"
    "No source or previous result modified.\n"
)
(out / "input_SHA256SUMS.txt").write_text(
    "\n".join(
        f"{hashlib.sha256(Path(os.environ[k]).read_bytes()).hexdigest()}  "
        f"{os.environ[k]}"
        for k in ["REVIEW", "SUBJECTS", "APARC", "ASEG"]
    ) + "\n"
)

print("\n===== BASELINE REPRODUCTION =====")
print("All fields reproduced:", all(r["reproduced"] for r in checks))
print("\n===== ORIGINAL ENRICHED REGION COUNTS =====")
print(baseline_selected.groupby("family").size().to_string())
print("\n===== AGE OVERLAP: PREVIOUSLY ENRICHED REGIONS =====")
print(tracked.loc[tracked["variant"].eq("AGE_OVERLAP")].to_string(index=False))
print("\nOUTPUT:", out)
