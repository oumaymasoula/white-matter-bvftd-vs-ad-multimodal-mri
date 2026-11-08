import os
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

source = Path(os.environ["INPUT"])
out = Path(os.environ["OUT"])
df = pd.read_csv(source, sep="\t")

def save(frame, name):
    frame.to_csv(out / "tables" / f"{name}.tsv", sep="\t", index=False)
    frame.to_csv(out / "tables" / f"{name}.csv", index=False)

if df.duplicated(["dataset", "subject"]).any():
    raise RuntimeError("Duplicate subject keys")

expected = {
    ("NIFD", "CN"): 67, ("NIFD", "bvFTD"): 42,
    ("ADNI", "CN"): 76, ("ADNI", "MCI"): 83, ("ADNI", "AD"): 39,
}
if df.groupby(["dataset", "group"]).size().to_dict() != expected:
    raise RuntimeError("Unexpected cohort")

for c in ["dataset", "subject", "group", "site", "sex",
          "protocol", "scanner_model"]:
    df[c] = df[c].astype(str).str.strip()

pairs = [
    ("frontal_forceps",
     "frontal_composite_thickness_mm", "FA_forceps_minor"),
    ("orbitofrontal_uncinate",
     "orbitofrontal_composite_thickness_mm", "FA_uncinate_bilateral"),
]
for c in ["age"] + [x for _, x, _ in pairs] + [y for _, _, y in pairs]:
    df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[c]).all():
        raise RuntimeError(f"Nonfinite: {c}")

diagnostics, counts, slope_rows = [], [], []

def fit_dataset(frame, dataset, variant):
    groups = ["CN", "bvFTD"] if dataset == "NIFD" else ["CN", "MCI", "AD"]
    cohort = frame.loc[frame["dataset"].eq(dataset)].copy()
    group_counts = cohort["group"].value_counts()
    for g in groups:
        counts.append({
            "variant": variant, "dataset": dataset, "group": g,
            "n": int(group_counts.get(g, 0))
        })

    output = {}
    for pair, structural, diffusion in pairs:
        status = "BLOCKED_GROUP_N_LT_5"
        fit = robust = None
        if all(group_counts.get(g, 0) >= 5 for g in groups):
            f = cohort.copy()
            means = f.groupby("group")[structural].mean()
            f["x"] = f[structural] - f["group"].map(means)
            f["y"] = f[diffusion]
            f["age_c"] = f["age"] - f["age"].mean()
            f["group"] = pd.Categorical(f["group"], categories=groups)

            formula = (
                "y ~ C(group, Treatment(reference='CN')) * x"
                " + age_c + C(sex) + C(site)"
            )
            formula += (
                " + C(protocol)" if dataset == "NIFD"
                else " + C(scanner_model)"
            )
            fit = smf.ols(formula, data=f, missing="raise").fit()
            X = np.asarray(fit.model.exog)
            rank = np.linalg.matrix_rank(X)
            p = X.shape[1]
            h = np.sum(X * np.linalg.pinv(X).T, axis=1)
            status = (
                "BLOCKED_RANK_DEFICIENT" if rank < p else
                "BLOCKED_NO_RESIDUAL_DF" if fit.df_resid <= 0 else
                "BLOCKED_NEAR_UNIT_LEVERAGE"
                if h.max() >= 1 - 1e-8 else
                "PASS_REVIEW_INFLUENCE"
            )
            if status == "PASS_REVIEW_INFLUENCE":
                robust = fit.get_robustcov_results(cov_type="HC3", use_t=True)
                if not np.isfinite(robust.cov_params()).all():
                    status = "BLOCKED_NONFINITE_HC3"
                    robust = None

            diagnostics.append({
                "variant": variant, "dataset": dataset, "pair": pair,
                "n": len(f), "parameters": p, "rank": rank,
                "residual_df": fit.df_resid,
                "max_leverage": float(h.max()),
                "formula": formula, "status": status,
            })
        else:
            diagnostics.append({
                "variant": variant, "dataset": dataset, "pair": pair,
                "n": len(cohort), "status": status,
            })

        for g in groups[1:]:
            b = se = low = high = pv = np.nan
            row_status = status
            if robust is not None:
                assigned = f.copy()
                assigned["group"] = pd.Categorical(
                    [g]*len(f), categories=groups
                )
                assigned["x"] = 0.0
                X0 = np.asarray(build_design_matrices(
                    [fit.model.data.design_info], assigned
                )[0])
                assigned["x"] = 1.0
                X1 = np.asarray(build_design_matrices(
                    [fit.model.data.design_info], assigned
                )[0])
                c = (X1-X0).mean(axis=0)
                variance = float(c @ robust.cov_params() @ c)
                if np.isfinite(variance) and variance > 0:
                    b = float(c @ robust.params)
                    se = math.sqrt(variance)
                    critical = stats.t.ppf(.975, fit.df_resid)
                    low, high = b-critical*se, b+critical*se
                    pv = 2*stats.t.sf(abs(b/se), fit.df_resid)
                else:
                    row_status = "BLOCKED_INVALID_VARIANCE"

            record = {
                "variant": variant, "dataset": dataset,
                "group": g, "pair": pair,
                "n_group": int(group_counts.get(g, 0)),
                "slope_FA_per_mm": b, "SE_HC3": se,
                "CI_low": low, "CI_high": high,
                "p_value": pv, "status": row_status,
            }
            output[(pair, g)] = record
            slope_rows.append(record)
    return output

def compare(nifd, adni, variant, omitted_dataset="", omitted_subject=""):
    records = []
    for pair, _, _ in pairs:
        nr = nifd[(pair, "bvFTD")]
        for g in ["AD", "MCI"]:
            ar = adni[(pair, g)]
            delta = se = low = high = pv = np.nan
            valid = np.isfinite([
                nr["slope_FA_per_mm"], nr["SE_HC3"],
                ar["slope_FA_per_mm"], ar["SE_HC3"]
            ]).all()
            if valid:
                delta = nr["slope_FA_per_mm"] - ar["slope_FA_per_mm"]
                se = math.hypot(nr["SE_HC3"], ar["SE_HC3"])
                z = stats.norm.ppf(.975)
                low, high = delta-z*se, delta+z*se
                pv = 2*stats.norm.sf(abs(delta/se))
            records.append({
                "variant": variant, "pair": pair,
                "comparison": f"bvFTD_slope_minus_{g}_slope",
                "bvFTD_slope_FA_per_mm": nr["slope_FA_per_mm"],
                "ADNI_slope_FA_per_mm": ar["slope_FA_per_mm"],
                "difference_FA_per_mm": delta,
                "SE_independent": se, "CI_low": low, "CI_high": high,
                "p_value": pv,
                "status": "PASS_REVIEW_INFLUENCE" if valid
                          else "BLOCKED_SOURCE_MODEL",
                "omitted_dataset": omitted_dataset,
                "omitted_subject": omitted_subject,
            })
    return records

baseline_n = fit_dataset(df, "NIFD", "BASELINE")
baseline_a = fit_dataset(df, "ADNI", "BASELINE")
cross_rows = compare(baseline_n, baseline_a, "BASELINE")

ranges = df.groupby(["dataset", "group"])["age"].agg(["min", "max"])
lower, upper = ranges["min"].max(), ranges["max"].min()
variants = {
    "AGE_OVERLAP": df.loc[df["age"].between(lower, upper)].copy()
}
definitions = [
    {"variant": "BASELINE", "definition": "Supported 307"},
    {"variant": "AGE_OVERLAP",
     "definition": f"Age {lower:.6f} to {upper:.6f} years"},
]

nifd = df.loc[df["dataset"].eq("NIFD")]
pc = pd.crosstab(nifd["protocol"], nifd["group"])
shared = pc.index[(pc["CN"] > 0) & (pc["bvFTD"] > 0)].tolist()
keep = ~df["dataset"].eq("NIFD") | df["protocol"].isin(shared)
variants["NIFD_SHARED_PROTOCOLS"] = df.loc[keep].copy()
definitions.append({
    "variant": "NIFD_SHARED_PROTOCOLS",
    "definition": "; ".join(shared)
})
largest = pc.loc[shared].sum(axis=1).idxmax()
keep = ~df["dataset"].eq("NIFD") | df["protocol"].eq(largest)
variants["NIFD_SINGLE_PROTOCOL"] = df.loc[keep].copy()
definitions.append({
    "variant": "NIFD_SINGLE_PROTOCOL", "definition": largest
})

for i, scanner in enumerate(sorted(
    df.loc[df["dataset"].eq("ADNI"), "scanner_model"].unique()
), start=1):
    name = f"ADNI_SCANNER_{i:02d}"
    keep = ~df["dataset"].eq("ADNI") | df["scanner_model"].eq(scanner)
    variants[name] = df.loc[keep].copy()
    definitions.append({"variant": name, "definition": scanner})

for name, frame in variants.items():
    n = fit_dataset(frame, "NIFD", name)
    a = fit_dataset(frame, "ADNI", name)
    cross_rows.extend(compare(n, a, name))

cross = pd.DataFrame(cross_rows)
cross["BH_q_four_comparisons"] = np.nan
cross["FDR_family_complete"] = False
for name, idx in cross.groupby("variant").groups.items():
    p = cross.loc[idx, "p_value"].to_numpy()
    if np.isfinite(p).all():
        cross.loc[idx, "BH_q_four_comparisons"] = multipletests(
            p, method="fdr_bh"
        )[1]
        cross.loc[idx, "FDR_family_complete"] = True

# Leave each participant out without choosing deletions by imaging results.
loo_rows = []
for i, subject in enumerate(df.itertuples(index=False), start=1):
    keep = ~(
        df["dataset"].eq(subject.dataset)
        & df["subject"].eq(subject.subject)
    )
    name = f"LOO_{i:03d}"
    refit = fit_dataset(df.loc[keep].copy(), subject.dataset, name)
    n = refit if subject.dataset == "NIFD" else baseline_n
    a = refit if subject.dataset == "ADNI" else baseline_a
    loo_rows.extend(compare(
        n, a, name, subject.dataset, subject.subject
    ))
    if i % 50 == 0:
        print(f"Completed {i}/{len(df)} participant omissions", flush=True)

loo = pd.DataFrame(loo_rows)
baseline = cross.loc[cross["variant"].eq("BASELINE"), [
    "pair", "comparison", "difference_FA_per_mm"
]].rename(columns={"difference_FA_per_mm": "baseline_difference"})
loo = loo.merge(baseline, on=["pair", "comparison"], validate="many_to_one")
loo["change_from_baseline"] = (
    loo["difference_FA_per_mm"] - loo["baseline_difference"]
)

summary = []
for (pair, comparison), part in loo.groupby(["pair", "comparison"]):
    valid = part.loc[part["difference_FA_per_mm"].notna()]
    baseline_value = float(part["baseline_difference"].iloc[0])
    summary.append({
        "pair": pair, "comparison": comparison,
        "baseline_difference": baseline_value,
        "omissions_attempted": len(part),
        "valid_omissions": len(valid),
        "blocked_omissions": len(part)-len(valid),
        "minimum_difference": valid["difference_FA_per_mm"].min(),
        "maximum_difference": valid["difference_FA_per_mm"].max(),
        "maximum_absolute_change": valid["change_from_baseline"].abs().max(),
        "direction_reversals": int((
            np.sign(valid["difference_FA_per_mm"])
            != np.sign(baseline_value)
        ).sum()),
    })

save(pd.DataFrame(slope_rows), "patient_slopes_native_units")
save(cross, "formal_slope_comparisons")
save(loo, "leave_one_participant_out")
save(pd.DataFrame(summary), "influence_summary")
save(pd.DataFrame(diagnostics), "model_diagnostics")
save(pd.DataFrame(counts), "group_counts")
save(pd.DataFrame(definitions), "variant_definitions")

(out / "README.txt").write_text(
    "Exploratory analysis of two previously selected cross-modal pairs.\n"
    "Slopes reported as FA per mm thickness, not per group SD.\n"
    "Within-dataset diagnosis-specific slopes with shared covariate effects.\n"
    "HC3 covariance and residual-df t inference for patient slopes.\n"
    "Cross-dataset slope differences: independent variances, normal inference.\n"
    "Four-comparison BH-FDR family separately within each named variant.\n"
    "Leave-one-participant-out runs are influence diagnostics, not "
    "independent replications or confirmatory significance tests.\n"
    "No preferred sample chosen using p-values.\n"
    "Age restriction is not matching; severity not adjusted.\n"
    "Cross-cohort acquisition confounding remains possible.\n"
    "No original measurements or previous results modified.\n"
)
(out / "input_SHA256SUMS.txt").write_text(
    f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {source}\n"
)
print("\n===== FORMAL SLOPE COMPARISONS =====")
print(cross.to_string(index=False))
print("\n===== INFLUENCE SUMMARY =====")
print(pd.DataFrame(summary).to_string(index=False))
print("\nOUTPUT:", out)
