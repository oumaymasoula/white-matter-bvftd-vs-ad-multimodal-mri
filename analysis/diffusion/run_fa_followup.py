"""Post hoc FA bootstrap, tract-set and age-68 sensitivity analyses."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
import run_fa_severity as base


def age68_fit(data, outcome, disease):
    age = data.age.to_numpy() - 68
    columns = {
        "intercept": np.ones(len(data)),
        "age68": age,
    }
    groups = ["bvFTD"] if disease == "bvFTD" else ["MCI", "AD"]

    for group in groups:
        indicator = data.group.eq(group).astype(float).to_numpy()
        columns[group] = indicator
        columns[group + "_age"] = indicator * age

    categories = [
        "sex", "site",
        "protocol" if disease == "bvFTD" else "scanner_model",
    ]
    for category in categories:
        if data[category].isna().any():
            raise ValueError("Missing " + category)
        dummy = pd.get_dummies(
            data[category].astype(str),
            prefix=category,
            drop_first=True,
            dtype=float,
        )
        for name in dummy:
            columns[name] = dummy[name].to_numpy()

    X = np.column_stack(list(columns.values()))
    inverse = np.linalg.pinv(X)
    rank = np.linalg.matrix_rank(X)
    df = len(data) - X.shape[1]
    leverage = (X * inverse.T).sum(axis=1)

    result = dict(
        status="blocked",
        n=len(data),
        parameters=X.shape[1],
        rank=int(rank),
        residual_df=df,
        max_leverage=float(leverage.max()),
        reason="",
    )
    patients = data[data.group.eq(disease)]
    controls = data[data.group.eq("CN")]

    if (
        min(len(patients), len(controls)) < 10
        or rank < X.shape[1]
        or df < 10
        or leverage.max() >= .99
    ):
        result["reason"] = "sample_rank_df_or_leverage"
        return result

    Y = data[outcome].to_numpy()
    beta = inverse @ Y
    residual = Y - X @ beta
    covariance = (
        inverse * (residual / (1 - leverage)) ** 2
    ) @ inverse.T

    index = list(columns).index(disease)
    variance = float(covariance[index, index])
    control_sd = float(controls[outcome].std(ddof=1))

    if (
        not np.isfinite(variance) or variance <= 0
        or not np.isfinite(control_sd) or control_sd <= 0
    ):
        result["reason"] = "invalid_variance"
        return result

    result.update(
        status="ok",
        beta=float(beta[index]),
        variance=variance,
        control_SD_scaled_effect=float(beta[index] / control_sd),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / "scratch/WM_Project/clinical_analysis",
    )
    parser.add_argument("--dti", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    if args.replicates < 200:
        raise ValueError("Use at least 200 replicates; 2000 for reporting.")

    clinical_path = args.root / "outputs/tables/clinical_imaging_307.tsv"
    dti_path = args.dti or base.locate_dti(
        args.root.parent / "statistical_analysis"
        / base.OLD / "input_SHA256SUMS.txt"
    )

    clinical = pd.read_csv(clinical_path, sep="\t")
    raw = pd.read_csv(dti_path, sep="\t")

    for frame in [clinical, raw]:
        for key in base.KEY:
            frame[key] = frame[key].astype(str).str.strip()
        if "group" in frame:
            frame["group"] = (
                frame.group.astype(str).str.strip()
                .replace({"Probable_AD": "AD"})
            )

    assert len(clinical) == 307
    assert not clinical.duplicated(base.KEY).any()
    assert clinical.groupby(["dataset", "group"]).size().to_dict() == base.EXPECTED
    assert not raw.duplicated(base.KEY + ["tract"]).any()

    selected = raw.merge(
        clinical[base.KEY + ["group"]],
        on=base.KEY,
        suffixes=("_source", ""),
        validate="many_to_one",
    )
    if "group_source" in selected:
        assert selected.group_source.eq(selected.group).all()

    assert (
        selected.primary_analysis_eligible.astype(str)
        .str.strip().str.lower()
        .isin(["yes", "true", "1", "1.0"]).all()
    )

    selected["FA_mean"] = pd.to_numeric(
        selected.FA_mean, errors="raise"
    )
    assert selected.FA_mean.between(0, 1).all()
    assert selected.groupby(base.KEY).size().eq(20).all()

    wide = selected.pivot(
        index=base.KEY, columns="tract", values="FA_mean"
    )
    assert wide.shape == (307, 20)
    assert wide.notna().all().all()

    data = clinical.merge(
        wide.reset_index(), on=base.KEY, validate="one_to_one"
    )
    labels = {base.normalize(name): name for name in wide.columns}

    def tract_name(key):
        if key not in labels:
            raise ValueError("Missing tract " + key)
        return labels[key]

    anterior = [
        tract_name(key) for key in [
            "forcepsminor",
            "uncinatefasciculusl",
            "uncinatefasciculusr",
        ]
    ]
    comparator = [
        tract_name(key) for key in [
            "forcepsmajor",
            "corticospinaltractl",
            "corticospinaltractr",
            "cingulumhippocampusl",
            "cingulumhippocampusr",
        ]
    ]

    data["FA_forceps_minor"] = data[tract_name("forcepsminor")]
    data["FA_uncinate_bilateral"] = (
        data[tract_name("uncinatefasciculusl")]
        + data[tract_name("uncinatefasciculusr")]
    ) / 2

    # Equal weighting of individual atlas tracts within each set.
    # This participant-level difference preserves within-person covariance.
    data["FA_anterior_minus_comparator"] = (
        data[anterior].mean(axis=1)
        - data[comparator].mean(axis=1)
    )

    targets = ["FA_forceps_minor", "FA_uncinate_bilateral"]
    cohorts = {
        dataset: data[data.dataset.eq(dataset)].reset_index(drop=True)
        for dataset in ["NIFD", "ADNI"]
    }
    expected = [
        (-.1113578129702013, -.0139138578268394),
        (-.07315245050689405, -.008636763792310494),
    ]

    original = {}
    for outcome, reference in zip(targets, expected):
        nifd = base.fit(cohorts["NIFD"], outcome, "bvFTD")
        adni = base.fit(cohorts["ADNI"], outcome, "AD")
        if nifd["status"] != "ok" or adni["status"] != "ok":
            raise RuntimeError("Baseline model blocked")

        assert np.allclose(
            [nifd["beta"], adni["beta"]],
            reference, rtol=1e-7, atol=1e-10,
        ), "Baseline reproduction failed"

        original[outcome] = (nifd, adni)

    print("Original FA coefficients reproduced successfully.", flush=True)

    output = args.root / "outputs" / (
        "FA_followup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)

    # One post hoc native-unit tract-set contrast.
    outcome = "FA_anterior_minus_comparator"
    nifd = base.fit(cohorts["NIFD"], outcome, "bvFTD")
    adni = base.fit(cohorts["ADNI"], outcome, "AD")

    tract_set = pd.DataFrame([
        dict(outcome=outcome, **base.compare(nifd, adni))
    ])
    tract_set.to_csv(
        output / "tract_set_contrast.tsv", sep="\t", index=False
    )

    diagnostics = [
        dict(analysis="tract_set", dataset=dataset, **model)
        for dataset, model in [("NIFD", nifd), ("ADNI", adni)]
    ]

    # Group-by-age effects at age 68 within observed age overlap.
    age_results = []
    age_counts = []

    for outcome in targets:
        fits = {}
        for dataset, disease in [("NIFD", "bvFTD"), ("ADNI", "AD")]:
            frame = cohorts[dataset]
            frame = frame[frame.age.between(61.6905, 74.8024)]
            frame = base.supported(frame, disease)

            fits[dataset] = age68_fit(frame, outcome, disease)
            diagnostics.append(dict(
                analysis="age68", outcome=outcome,
                dataset=dataset, **fits[dataset],
            ))

            if outcome == targets[0]:
                for group, subset in frame.groupby("group"):
                    age_counts.append(dict(
                        dataset=dataset,
                        group=group,
                        n=len(subset),
                        mean_age=subset.age.mean(),
                        min_age=subset.age.min(),
                        max_age=subset.age.max(),
                    ))

        age_results.append(dict(
            outcome=outcome,
            **base.compare(fits["NIFD"], fits["ADNI"]),
        ))

    age = pd.DataFrame(age_results)
    age["q_two_FA_age68"] = np.nan

    if age.status.eq("ok").all():
        age["q_two_FA_age68"] = stats.false_discovery_control(
            age.p.to_numpy(), method="bh"
        )

    age.to_csv(
        output / "FA_age68_interaction.tsv", sep="\t", index=False
    )
    pd.DataFrame(age_counts).to_csv(
        output / "FA_age68_group_counts.tsv", sep="\t", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        output / "model_diagnostics.tsv", sep="\t", index=False
    )

    # Participant pairs bootstrap within diagnosis-by-site strata.
    # Both outcomes travel together with each sampled participant.
    # Models and cohort-specific control SDs are recomputed each time.
    rng = np.random.default_rng(args.seed)
    strata = {
        dataset: [
            subset.index.to_numpy()
            for _, subset in frame.groupby(
                ["group", "site"], dropna=False
            )
        ]
        for dataset, frame in cohorts.items()
    }

    draws = []
    for replicate in range(args.replicates):
        sampled = {}
        for dataset, frame in cohorts.items():
            indices = np.concatenate([
                rng.choice(ids, len(ids), replace=True)
                for ids in strata[dataset]
            ])
            sampled[dataset] = frame.iloc[indices].reset_index(drop=True)

        for outcome in targets:
            nifd = base.fit(sampled["NIFD"], outcome, "bvFTD")
            adni = base.fit(sampled["ADNI"], outcome, "AD")
            row = dict(
                replicate=replicate, outcome=outcome, status="blocked"
            )

            if nifd["status"] == "ok" and adni["status"] == "ok":
                row.update(
                    status="ok",
                    NIFD_scaled=nifd["control_SD_scaled_effect"],
                    ADNI_scaled=adni["control_SD_scaled_effect"],
                    difference=(
                        nifd["control_SD_scaled_effect"]
                        - adni["control_SD_scaled_effect"]
                    ),
                )
            draws.append(row)

        if (replicate + 1) % 200 == 0:
            print(
                "Bootstrap replicates:", replicate + 1, flush=True
            )

    draws = pd.DataFrame(draws)
    draws.to_csv(
        output / "bootstrap_draws.tsv", sep="\t", index=False
    )

    summary = []
    for outcome in targets:
        valid = draws[
            draws.outcome.eq(outcome) & draws.status.eq("ok")
        ]
        nifd, adni = original[outcome]

        estimates = [
            ("NIFD_scaled", nifd["control_SD_scaled_effect"]),
            ("ADNI_scaled", adni["control_SD_scaled_effect"]),
            (
                "difference",
                nifd["control_SD_scaled_effect"]
                - adni["control_SD_scaled_effect"],
            ),
        ]

        for field, point in estimates:
            acceptable = len(valid) >= .95 * args.replicates
            low, high = (
                np.quantile(valid[field], [.025, .975])
                if len(valid) else [np.nan, np.nan]
            )
            summary.append(dict(
                outcome=outcome,
                estimand=field,
                point=point,
                valid_replicates=len(valid),
                attempted=args.replicates,
                CI_low=low if acceptable else np.nan,
                CI_high=high if acceptable else np.nan,
                status=(
                    "ok" if acceptable
                    else "blocked_over_5pct_failed"
                ),
            ))

    summary = pd.DataFrame(summary)
    summary.to_csv(
        output / "control_SD_bootstrap_CI.tsv",
        sep="\t", index=False,
    )

    metadata = dict(
        baseline_reproduction="passed",
        replicates=args.replicates,
        seed=args.seed,
        anterior_tracts=anterior,
        comparator_tracts=comparator,
        bootstrap=(
            "Diagnosis-by-site stratified participant pairs; "
            "control SD and adjusted coefficients refitted each draw; "
            "percentile intervals; failure proportion reported"
        ),
        tract_set=(
            "One post hoc native-unit patient-level "
            "anterior-minus-comparator outcome; no posterior gradient "
            "or independent validation claimed"
        ),
        age68=(
            "Group-by-age model on observed overlap; "
            "two-outcome exploratory BH family; "
            "approximate Welch HC3 inference"
        ),
        limits=(
            "No acquisition harmonization or disease-by-scanner test. "
            "Percentile intervals are marginal, not multiplicity-adjusted. "
            "Sparse site strata condition bootstrap uncertainty "
            "on observed sites."
        ),
        SHA256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                clinical_path, dti_path,
                Path(__file__), Path(base.__file__),
            ]
        },
    )
    (output / "analysis_provenance.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nTRACT-SET CONTRAST")
    print(tract_set.to_string(index=False))
    print("\nAGE-68 CONTRASTS")
    print(age.to_string(index=False))
    print("\nCONTROL-SD BOOTSTRAP")
    print(summary.to_string(index=False))
    print("\nOutputs:", output)


if __name__ == "__main__":
    main()
