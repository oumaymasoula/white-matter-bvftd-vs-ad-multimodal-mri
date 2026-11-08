import os
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import scipy
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

source = Path(os.environ["INPUT"])
out = Path(os.environ["OUT"])
df = pd.read_csv(source, sep="\t")

def save(frame, name):
    frame.to_csv(out / "tables" / f"{name}.tsv",
                 sep="\t", index=False)
    frame.to_csv(out / "tables" / f"{name}.csv", index=False)

if df.duplicated(["dataset", "subject"]).any():
    raise RuntimeError("Duplicate participant keys")

expected = {
    ("NIFD", "CN"): 67, ("NIFD", "bvFTD"): 42,
    ("ADNI", "CN"): 76, ("ADNI", "MCI"): 83, ("ADNI", "AD"): 39,
}
if df.groupby(["dataset", "group"]).size().to_dict() != expected:
    raise RuntimeError("Unexpected participant counts")

pairs = [
    ("frontal_forceps",
     "frontal_composite_thickness_mm", "FA_forceps_minor"),
    ("orbitofrontal_uncinate",
     "orbitofrontal_composite_thickness_mm", "FA_uncinate_bilateral"),
]

results, diagnostics, influence, scaling = [], [], [], []

for dataset, groups in [
    ("NIFD", ["CN", "bvFTD"]),
    ("ADNI", ["CN", "MCI", "AD"]),
]:
    for pair, structural, diffusion in pairs:
        frame = df.loc[df["dataset"].eq(dataset)].copy()

        for column in ["site", "sex", "protocol", "scanner_model"]:
            frame[column] = frame[column].astype(str).str.strip()

        for column in ["age", structural, diffusion]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column]).all():
                raise RuntimeError(f"Nonfinite values: {column}")

        group_means = frame.groupby("group")[structural].mean()
        group_sds = frame.groupby("group")[structural].std(ddof=1)

        # Center thickness within diagnosis without changing its mm units.
        frame["x"] = (
            frame[structural] - frame["group"].map(group_means)
        )
        frame["y"] = frame[diffusion]
        frame["age_c"] = frame["age"] - frame["age"].mean()
        frame["group"] = pd.Categorical(
            frame["group"], categories=groups
        )

        formula = (
            "y ~ C(group, Treatment(reference='CN')) * x"
            " + age_c + C(sex) + C(site)"
        )
        formula += (
            " + C(protocol)" if dataset == "NIFD"
            else " + C(scanner_model)"
        )

        fit = smf.ols(formula, data=frame, missing="raise").fit()
        X = np.asarray(fit.model.exog)
        rank = np.linalg.matrix_rank(X)
        parameters = X.shape[1]
        leverage = np.sum(X * np.linalg.pinv(X).T, axis=1)

        status = (
            "BLOCKED_RANK_DEFICIENT" if rank < parameters else
            "BLOCKED_NO_RESIDUAL_DF" if fit.df_resid <= 0 else
            "BLOCKED_NEAR_UNIT_LEVERAGE"
            if leverage.max() >= 1 - 1e-8 else
            "PASS_REVIEW_INFLUENCE"
        )

        robust = None
        if status == "PASS_REVIEW_INFLUENCE":
            robust = fit.get_robustcov_results(
                cov_type="HC3", use_t=True
            )
            if not np.isfinite(robust.cov_params()).all():
                status = "BLOCKED_NONFINITE_HC3"
                robust = None

        diagnostics.append({
            "dataset": dataset, "pair": pair, "n": len(frame),
            "formula": formula, "parameters": parameters,
            "rank": rank, "residual_df": fit.df_resid,
            "max_leverage": float(leverage.max()),
            "condition_number": float(fit.condition_number),
            "status": status,
        })

        if robust is not None:
            cook = fit.get_influence().cooks_distance[0]
            for subject, hi, ci in zip(
                frame["subject"], leverage, cook
            ):
                influence.append({
                    "dataset": dataset, "pair": pair,
                    "subject": subject, "leverage": hi,
                    "Cooks_D": ci,
                    "Cooks_flag_4_over_n": bool(ci > 4/len(frame)),
                })

        for group in groups[1:]:
            sd = float(group_sds[group])
            scaling.append({
                "dataset": dataset, "group": group,
                "pair": pair, "thickness_mean_mm": group_means[group],
                "thickness_SD_mm": sd,
            })

            beta = se = low = high = pvalue = np.nan
            row_status = status

            if not np.isfinite(sd) or sd <= 0:
                row_status = "BLOCKED_CONSTANT_PREDICTOR"
            elif robust is not None:
                # Design contrast for the group's thickness slope.
                assigned = frame.copy()
                assigned["group"] = pd.Categorical(
                    [group] * len(frame), categories=groups
                )
                assigned["x"] = 0.0
                X0 = np.asarray(build_design_matrices(
                    [fit.model.data.design_info], assigned
                )[0])
                assigned["x"] = 1.0
                X1 = np.asarray(build_design_matrices(
                    [fit.model.data.design_info], assigned
                )[0])

                contrast = (X1 - X0).mean(axis=0) * sd
                covariance = np.asarray(robust.cov_params())
                beta = float(contrast @ robust.params)
                variance = float(contrast @ covariance @ contrast)

                if np.isfinite(variance) and variance > 0:
                    se = np.sqrt(variance)
                    critical = stats.t.ppf(.975, fit.df_resid)
                    low, high = beta-critical*se, beta+critical*se
                    pvalue = 2*stats.t.sf(
                        abs(beta/se), fit.df_resid
                    )
                else:
                    beta = np.nan
                    row_status = "BLOCKED_INVALID_CONTRAST_VARIANCE"

            results.append({
                "dataset": dataset, "group": group, "pair": pair,
                "n_group": int(frame["group"].eq(group).sum()),
                "n_joint_model": len(frame),
                "thickness_predictor": structural,
                "FA_outcome": diffusion,
                "estimate_FA_per_group_SD_thickness": beta,
                "SE_HC3": se, "CI_low": low, "CI_high": high,
                "p_value": pvalue, "status": row_status,
            })

result = pd.DataFrame(results)
if len(result) != 6:
    raise RuntimeError("Expected exactly six patient-group tests")

result["BH_q_six_tests"] = np.nan
result["FDR_family_complete"] = False
if np.isfinite(result["p_value"]).all():
    result["BH_q_six_tests"] = multipletests(
        result["p_value"], method="fdr_bh"
    )[1]
    result["FDR_family_complete"] = True

save(result, "joint_patient_slopes")
save(pd.DataFrame(diagnostics), "joint_model_diagnostics")
save(pd.DataFrame(scaling), "thickness_scaling")
save(pd.DataFrame(influence), "joint_influence_audit")

(out / "README.txt").write_text(
    "Exploratory cross-modal associations in the supported 307 cohort.\n"
    "Two pairs, three patient groups: six-test BH-FDR family.\n"
    "Models fitted separately within NIFD and ADNI.\n"
    "Diagnosis-specific thickness slopes; shared age, sex, site and "
    "acquisition effects within each dataset.\n"
    "Thickness centered within diagnosis; reported slopes multiplied "
    "by the corresponding patient group's thickness SD.\n"
    "HC3 covariance and residual-df t inference.\n"
    "No participant deletion based on imaging outcomes.\n"
    "Blocked models are not interpreted as negative associations.\n"
    "A significant slope in one group and nonsignificant slope in "
    "another does not establish a difference between their slopes.\n"
    "No causal, mechanistic or disease-specific signature claim.\n"
    "No previous result modified.\n"
)
(out / "input_SHA256SUMS.txt").write_text(
    f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {source}\n"
)
(out / "software_versions.txt").write_text(
    f"Python={platform.python_version()}\n"
    f"NumPy={np.__version__}\npandas={pd.__version__}\n"
    f"SciPy={scipy.__version__}\n"
    f"statsmodels={statsmodels.__version__}\n"
)

print("===== JOINT MODEL DIAGNOSTICS =====")
print(pd.DataFrame(diagnostics).to_string(index=False))
print("\n===== PATIENT-GROUP ASSOCIATIONS =====")
print(result.to_string(index=False))
print("\nOUTPUT:", out)
