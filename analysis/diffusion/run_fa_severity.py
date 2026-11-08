"""FA severity sensitivity preserving original diagnostic groups and covariates."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["dataset", "subject"]
OLD = "COMMON02_full_vs_supported_multimodal_20260922_211319_H9MfYZ"
EXPECTED = {
    ("NIFD", "bvFTD"): 42,
    ("NIFD", "CN"): 67,
    ("ADNI", "AD"): 39,
    ("ADNI", "MCI"): 83,
    ("ADNI", "CN"): 76,
}


def normalize(value):
    return re.sub("[^a-z0-9]", "", str(value).lower())


def locate_dti(manifest):
    matches = []
    for line in manifest.read_text().splitlines():
        bits = line.split(maxsplit=1)
        if len(bits) != 2:
            continue
        path = Path(bits[1].strip())
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as f:
            columns = next(csv.reader(f, delimiter="\t"))
        if {"dataset", "subject", "tract", "FA_mean"} <= set(columns):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != bits[0]:
                raise RuntimeError(f"Original DTI hash changed: {path}")
            matches.append(path)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one original DTI input; found {matches}. "
            "Use --dti with the verified source."
        )
    return matches[0]


def fit(frame, outcome, disease):
    d = frame.copy()
    result = {
        "status": "blocked",
        "n": len(d),
        "n_patient": int(d.group.eq(disease).sum()),
        "n_CN": int(d.group.eq("CN").sum()),
        "mean_patient_age": float(d.loc[d.group.eq(disease), "age"].mean()),
        "mean_control_age": float(d.loc[d.group.eq("CN"), "age"].mean()),
    }
    if min(result["n_patient"], result["n_CN"]) < 10:
        result["reason"] = "fewer_than_10_patients_or_controls"
        return result

    columns = {
        "intercept": np.ones(len(d)),
        "age_c": d.age.to_numpy() - d.age.mean(),
    }
    groups = ["bvFTD"] if disease == "bvFTD" else ["MCI", "AD"]
    for group in groups:
        columns[group] = d.group.eq(group).astype(float).to_numpy()

    categories = [
        "sex", "site",
        "protocol" if disease == "bvFTD" else "scanner_model",
    ]
    for category in categories:
        if d[category].isna().any():
            raise RuntimeError(f"Missing {category}")
        dummy = pd.get_dummies(
            d[category].astype(str),
            prefix=category,
            drop_first=True,
            dtype=float,
        )
        for name in dummy:
            columns[name] = dummy[name].to_numpy()

    X = np.column_stack(list(columns.values()))
    Y = d[outcome].to_numpy()
    rank = np.linalg.matrix_rank(X)
    inverse = np.linalg.pinv(X)
    leverage = (X * inverse.T).sum(axis=1)
    residual_df = len(d) - X.shape[1]

    result.update(
        rank=int(rank),
        parameters=X.shape[1],
        residual_df=residual_df,
        max_leverage=float(leverage.max()),
    )
    if rank < X.shape[1] or residual_df < 10 or leverage.max() >= .99:
        result["reason"] = "rank_residual_df_or_leverage"
        return result

    beta = inverse @ Y
    residual = Y - X @ beta
    covariance = (
        inverse * (residual / (1 - leverage)) ** 2
    ) @ inverse.T

    index = list(columns).index(disease)
    control_sd = float(d.loc[d.group.eq("CN"), outcome].std(ddof=1))
    variance = float(covariance[index, index])

    if (
        not np.isfinite(variance) or variance <= 0
        or not np.isfinite(control_sd) or control_sd <= 0
    ):
        result["reason"] = "invalid_variance_or_control_SD"
        return result

    se = variance ** .5
    critical = stats.t.ppf(.975, residual_df)
    result.update(
        status="ok",
        reason="",
        beta=float(beta[index]),
        variance=variance,
        SE_HC3=se,
        CI_low=float(beta[index] - critical * se),
        CI_high=float(beta[index] + critical * se),
        control_SD=control_sd,
        control_SD_scaled_effect=float(beta[index] / control_sd),
    )
    return result


def compare(nifd, adni):
    if nifd["status"] != "ok" or adni["status"] != "ok":
        return {"status": "blocked", "reason": "source_model_blocked"}

    v1, v2 = nifd["variance"], adni["variance"]
    se = (v1 + v2) ** .5
    df = (v1 + v2) ** 2 / (
        v1 * v1 / nifd["residual_df"]
        + v2 * v2 / adni["residual_df"]
    )
    estimate = nifd["beta"] - adni["beta"]
    critical = stats.t.ppf(.975, df)

    return {
        "status": "ok",
        "reason": "",
        "estimate": estimate,
        "SE_independent": se,
        "df_Welch_approx": df,
        "CI_low": estimate - critical * se,
        "CI_high": estimate + critical * se,
        "p": 2 * stats.t.sf(abs(estimate / se), df),
        "standardized_point_difference": (
            nifd["control_SD_scaled_effect"]
            - adni["control_SD_scaled_effect"]
        ),
    }


def supported(frame, disease):
    counts = frame.groupby(["site", "group"]).size().unstack(fill_value=0)
    keep = counts.index[
        (counts.get("CN", 0) > 0) & (counts.get(disease, 0) > 0)
    ]
    return frame[frame.site.isin(keep)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / "scratch/WM_Project/clinical_analysis",
    )
    parser.add_argument("--dti", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    clinical_path = args.root / "outputs/tables/clinical_imaging_307.tsv"
    dti_path = args.dti or locate_dti(
        args.root.parent / "statistical_analysis" / OLD / "input_SHA256SUMS.txt"
    )

    clinical = pd.read_csv(clinical_path, sep="\t")
    raw = pd.read_csv(dti_path, sep="\t")

    for frame in [clinical, raw]:
        for column in KEY:
            frame[column] = frame[column].astype(str).str.strip()
        if "group" in frame:
            frame["group"] = (
                frame["group"].astype(str).str.strip()
                .replace({"Probable_AD": "AD"})
            )

    assert len(clinical) == 307 and not clinical.duplicated(KEY).any()
    assert clinical.groupby(["dataset", "group"]).size().to_dict() == EXPECTED
    assert not raw.duplicated(KEY + ["tract"]).any()

    selected = raw.merge(
        clinical[KEY + ["group"]],
        on=KEY,
        suffixes=("_source", ""),
        validate="many_to_one",
    )
    if "group_source" in selected:
        assert selected.group_source.eq(selected.group).all()
    if "primary_analysis_eligible" in selected:
        assert selected.primary_analysis_eligible.astype(str).str.lower().isin(
            ["true", "1", "1.0", "yes"]
        ).all()

    selected["FA_mean"] = pd.to_numeric(selected.FA_mean, errors="raise")
    assert selected.FA_mean.between(0, 1).all()
    assert selected.groupby(KEY).size().eq(20).all()

    pivot = selected.pivot(index=KEY, columns="tract", values="FA_mean")
    assert len(pivot) == 307 and pivot.shape[1] == 20
    assert pivot.notna().all().all()

    names = {normalize(name): name for name in pivot.columns}

    def tract_name(key):
        if key not in names:
            raise RuntimeError(
                f"Tract label {key} missing; labels are {list(pivot.columns)}"
            )
        return names[key]

    forceps = tract_name("forcepsminor")
    left = tract_name("uncinatefasciculusl")
    right = tract_name("uncinatefasciculusr")

    data = clinical.merge(pivot.reset_index(), on=KEY, validate="one_to_one")
    data["FA_forceps_minor"] = data[forceps]
    data["FA_uncinate_bilateral"] = (data[left] + data[right]) / 2
    targets = ["FA_forceps_minor", "FA_uncinate_bilateral"]

    expected = {
        "FA_forceps_minor": (-.1113578129702013, -.0139138578268394),
        "FA_uncinate_bilateral": (-.07315245050689405, -.008636763792310494),
    }

    for outcome in targets:
        nifd = fit(data[data.dataset.eq("NIFD")], outcome, "bvFTD")
        adni = fit(data[data.dataset.eq("ADNI")], outcome, "AD")
        if nifd["status"] != "ok" or adni["status"] != "ok":
            raise RuntimeError("Unrestricted model blocked")
        if not np.allclose(
            [nifd["beta"], adni["beta"]],
            expected[outcome],
            rtol=1e-7,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Unrestricted reproduction failed for {outcome}: "
                f'{nifd["beta"]}, {adni["beta"]}; '
                "do not report sensitivities"
            )

    output = args.out or args.root / "outputs" / (
        "FA_severity_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)

    models, comparisons, counts = [], [], []
    variants = [
        "unrestricted",
        "clinical_complete",
        "CDR_0.5_1",
        "CDR_plus_MMSE",
        "CDR_MMSE_common_age",
    ]

    for variant in variants:
        frames = {}
        for dataset, disease in [("NIFD", "bvFTD"), ("ADNI", "AD")]:
            frame = data[data.dataset.eq(dataset)].copy()
            patient = frame.group.eq(disease)
            valid = frame.CDR_global.notna() & frame.MMSE.notna()

            if variant != "unrestricted":
                frame = frame[~patient | valid]
            if variant in variants[2:]:
                frame = frame[
                    ~frame.group.eq(disease)
                    | frame.CDR_global.between(.5, 1)
                ]
            if variant in variants[3:]:
                frame = frame[
                    ~frame.group.eq(disease)
                    | frame.MMSE.between(20, 26)
                ]
            if variant == "CDR_MMSE_common_age":
                frame = frame[frame.age.between(61.6905, 74.8024)]

            frame = supported(frame, disease)
            frames[dataset] = frame

            for group, subset in frame.groupby("group"):
                counts.append({
                    "variant": variant,
                    "dataset": dataset,
                    "group": group,
                    "n": len(subset),
                    "mean_age": subset.age.mean(),
                    "median_MMSE": subset.MMSE.median(),
                    "median_CDR": subset.CDR_global.median(),
                })

        for outcome in targets:
            nifd = fit(frames["NIFD"], outcome, "bvFTD")
            adni = fit(frames["ADNI"], outcome, "AD")
            models.extend([
                dict(variant=variant, outcome=outcome, dataset="NIFD", **nifd),
                dict(variant=variant, outcome=outcome, dataset="ADNI", **adni),
            ])
            comparisons.append(
                dict(variant=variant, outcome=outcome, **compare(nifd, adni))
            )

    results = pd.DataFrame(comparisons)
    results["q_two_FA"] = np.nan
    results["family_complete"] = False

    for _, indices in results.groupby("variant").groups.items():
        if results.loc[indices, "status"].eq("ok").all():
            results.loc[indices, "q_two_FA"] = stats.false_discovery_control(
                results.loc[indices, "p"].to_numpy(), method="bh"
            )
            results.loc[indices, "family_complete"] = True

    pd.DataFrame(models).to_csv(
        output / "FA_severity_model_estimates.tsv", sep="\t", index=False
    )
    pd.DataFrame(counts).to_csv(
        output / "FA_severity_group_counts.tsv", sep="\t", index=False
    )
    results.to_csv(
        output / "FA_severity_cross_disease.tsv", sep="\t", index=False
    )

    baseline = []
    for tract in pivot.columns:
        nifd = fit(data[data.dataset.eq("NIFD")], tract, "bvFTD")
        adni = fit(data[data.dataset.eq("ADNI")], tract, "AD")
        baseline.append(dict(
            tract=tract,
            NIFD_effect=nifd.get("beta"),
            ADNI_effect=adni.get("beta"),
            NIFD_control_SD=nifd.get("control_SD"),
            ADNI_control_SD=adni.get("control_SD"),
            **compare(nifd, adni),
        ))

    pd.DataFrame(baseline).to_csv(
        output / "all_20_FA_native_and_control_SD_points.tsv",
        sep="\t", index=False,
    )

    metadata = {
        "clinical_file": str(clinical_path),
        "DTI_file": str(dti_path),
        "baseline_reproduction": "passed",
        "SHA256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [clinical_path, dti_path]
        },
        "inference": (
            "Approximate Welch t using independent HC3 variances "
            "and model residual df"
        ),
        "FA_family": (
            "Two outcomes per variant; failed models block "
            "complete-family q values"
        ),
        "severity_complete_cases": (
            "MMSE and global CDR required for patient groups; "
            "MCI and controls otherwise retained"
        ),
        "standardization": (
            "Adjusted effect divided by retained cohort control sample SD. "
            "Descriptive point estimates only; no scaled CI or p value."
        ),
        "limitations": (
            "Age covariate adjustment is not joint age-severity matching; "
            "acquisition confounding remains."
        ),
    }
    (output / "analysis_provenance.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(results.to_string(index=False))
    print("\nOutputs:", output)


if __name__ == "__main__":
    main()
