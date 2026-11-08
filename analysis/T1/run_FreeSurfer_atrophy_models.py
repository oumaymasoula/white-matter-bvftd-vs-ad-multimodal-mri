import math
import os
import platform
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from patsy import build_design_matrices

input_path = Path(os.environ["INPUT"])
out = Path(os.environ["OUT"])

table_dir = out / "tables"
png_dir = out / "figures" / "png"
pdf_dir = out / "figures" / "pdf"
svg_dir = out / "figures" / "svg"

df = pd.read_csv(
    input_path,
    sep="\t",
    dtype={
        "dataset": str,
        "group": str,
        "subject": str,
    },
)

if len(df) != 343:
    raise RuntimeError(
        f"Expected 343 subjects; found {len(df)}"
    )


def locate(fields, candidates):
    for candidate in candidates:
        if candidate in fields:
            return candidate

    raise RuntimeError(
        "Could not identify required column from: "
        + ", ".join(candidates)
        + "\nAvailable columns: "
        + ", ".join(fields)
    )


fields = list(df.columns)

age_field = locate(
    fields,
    [
        "age_at_DWI_years",
        "age",
        "Age",
        "AGE",
    ],
)

sex_field = locate(
    fields,
    [
        "sex",
        "Sex",
        "SEX",
        "PTGENDER",
    ],
)

site_field = locate(
    fields,
    [
        "acquisition_site",
        "site",
        "SITE",
        "SITEID",
    ],
)

required_groups = {
    "ADNI": ["CN", "MCI", "AD"],
    "NIFD": ["CN", "bvFTD"],
}

expected_full_counts = {
    ("ADNI", "CN"): 81,
    ("ADNI", "MCI"): 101,
    ("ADNI", "AD"): 48,
    ("NIFD", "CN"): 68,
    ("NIFD", "bvFTD"): 45,
}

observed_counts = Counter(
    zip(df["dataset"], df["group"])
)

for key, expected in expected_full_counts.items():
    observed = observed_counts.get(key, 0)

    if observed != expected:
        raise RuntimeError(
            f"Unexpected count for {key}: "
            f"expected {expected}, found {observed}"
        )

unexpected = [
    key
    for key in observed_counts
    if key not in expected_full_counts
]

if unexpected:
    raise RuntimeError(
        f"Unexpected dataset/group combinations: {unexpected}"
    )

df["age_model"] = pd.to_numeric(
    df[age_field],
    errors="coerce",
)

df["eTIV_model"] = pd.to_numeric(
    df["eTIV_ml"],
    errors="coerce",
)

df["sex_model"] = (
    df[sex_field]
    .astype(str)
    .str.strip()
)

df["site_model"] = (
    df[site_field]
    .astype(str)
    .str.strip()
)

for field in [
    "age_model",
    "eTIV_model",
]:
    if not np.isfinite(df[field]).all():
        raise RuntimeError(
            f"Missing or invalid covariate: {field}"
        )

for field in [
    "sex_model",
    "site_model",
]:
    invalid = df[field].isin(
        ["", "nan", "None", "NA"]
    )

    if invalid.any():
        raise RuntimeError(
            f"Missing values in covariate: {field}"
        )

primary_features = [
    "hippocampus_volume_ml",
    "medial_temporal_composite_thickness_mm",
    "frontal_composite_thickness_mm",
    "temporal_composite_thickness_mm",
    "anterior_cingulate_composite_thickness_mm",
    "insula_thickness_mm",
    "lateral_ventricle_volume_ml",
]

secondary_features = [
    "cortical_gray_volume_ml",
    "cerebral_white_matter_volume_ml",
    "subcortical_gray_volume_ml",
    "amygdala_volume_ml",
    "corpus_callosum_volume_ml",
    "bilateral_MeanThickness_mm",
    "orbitofrontal_composite_thickness_mm",
]

all_features = primary_features + secondary_features

volume_features = {
    "hippocampus_volume_ml",
    "lateral_ventricle_volume_ml",
    "cortical_gray_volume_ml",
    "cerebral_white_matter_volume_ml",
    "subcortical_gray_volume_ml",
    "amygdala_volume_ml",
    "corpus_callosum_volume_ml",
}

for feature in all_features:
    if feature not in df.columns:
        raise RuntimeError(
            f"Missing outcome: {feature}"
        )

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

    if not np.isfinite(df[feature]).all():
        raise RuntimeError(
            f"Invalid values in outcome: {feature}"
        )

# Determine sites with representation from every group
# required for each dataset.
shared_sites = {}
site_rows = []

for dataset, groups in required_groups.items():
    subset = df[df["dataset"] == dataset]

    counts = pd.crosstab(
        subset["site_model"],
        subset["group"],
    )

    for group in groups:
        if group not in counts.columns:
            counts[group] = 0

    supported = []

    for site, row in counts.iterrows():
        is_supported = all(
            int(row[group]) > 0
            for group in groups
        )

        if is_supported:
            supported.append(site)

        for group in groups:
            site_rows.append(
                {
                    "dataset": dataset,
                    "site": site,
                    "group": group,
                    "n": int(row[group]),
                    "shared_site_status": (
                        "SUPPORTED"
                        if is_supported
                        else "UNSUPPORTED"
                    ),
                }
            )

    if not supported:
        raise RuntimeError(
            f"No shared sites found for {dataset}"
        )

    shared_sites[dataset] = sorted(supported)

pd.DataFrame(site_rows).to_csv(
    table_dir / "FreeSurfer_site_group_support.tsv",
    sep="\t",
    index=False,
)

analysis_specs = []

for dataset in ["ADNI", "NIFD"]:
    dataset_df = df[
        df["dataset"] == dataset
    ].copy()

    primary_df = dataset_df[
        dataset_df["site_model"].isin(
            shared_sites[dataset]
        )
    ].copy()

    analysis_specs.append(
        (
            "PRIMARY_SHARED_SITES",
            dataset,
            primary_df,
        )
    )

    analysis_specs.append(
        (
            "SENSITIVITY_FULL_COHORT",
            dataset,
            dataset_df,
        )
    )

membership_rows = []

for _, row in df.iterrows():
    dataset = row["dataset"]

    membership_rows.append(
        {
            "dataset": dataset,
            "group": row["group"],
            "subject": row["subject"],
            "site": row["site_model"],
            "primary_shared_site": (
                "YES"
                if row["site_model"]
                in shared_sites[dataset]
                else "NO"
            ),
            "full_cohort": "YES",
        }
    )

pd.DataFrame(membership_rows).to_csv(
    table_dir / "FreeSurfer_analysis_membership.tsv",
    sep="\t",
    index=False,
)

cohort_summary_rows = []

for analysis, dataset, work in analysis_specs:
    counts = Counter(work["group"])

    cohort_summary_rows.append(
        {
            "analysis": analysis,
            "dataset": dataset,
            "n_total": len(work),
            "group_counts": ";".join(
                f"{group}={counts.get(group, 0)}"
                for group in required_groups[dataset]
            ),
            "n_sites": work["site_model"].nunique(),
            "sites": ";".join(
                sorted(work["site_model"].unique())
            ),
        }
    )

pd.DataFrame(cohort_summary_rows).to_csv(
    table_dir / "FreeSurfer_analysis_cohort_summary.tsv",
    sep="\t",
    index=False,
)


def bh_fdr(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)

    finite = np.isfinite(values)
    p = values[finite]

    if not len(p):
        return result

    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)

    adjusted = (
        ranked
        * m
        / np.arange(1, m + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(adjusted, 1.0)

    restored = np.empty(m)
    restored[order] = adjusted
    result[finite] = restored

    return result


def vector_for_contrast(model, group_a, group_b):
    names = list(model.params.index)
    vector = np.zeros(len(names))

    def add_group(group, weight):
        if group == "CN":
            return

        matches = [
            index
            for index, name in enumerate(names)
            if name.startswith("C(group")
            and f"[T.{group}]" in name
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not identify coefficient for "
                f"group {group}: {names}"
            )

        vector[matches[0]] += weight

    add_group(group_a, 1.0)
    add_group(group_b, -1.0)

    return vector


def contrast_result(model, vector):
    test = model.t_test(vector)

    effect = float(
        np.asarray(test.effect).reshape(-1)[0]
    )

    standard_error = float(
        np.asarray(test.sd).reshape(-1)[0]
    )

    p_value = float(
        np.asarray(test.pvalue).reshape(-1)[0]
    )

    interval = np.asarray(
        test.conf_int(alpha=0.05)
    ).reshape(-1)

    return {
        "estimate": effect,
        "standard_error": standard_error,
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "p_HC3": p_value,
    }


def marginal_group_estimate(model, work, group):
    new = work.copy()
    new["group"] = group

    design_info = model.model.data.design_info

    design = np.asarray(
        build_design_matrices(
            [design_info],
            new,
            return_type="dataframe",
        )[0],
        dtype=float,
    )

    vector = design.mean(axis=0)
    beta = np.asarray(model.params, dtype=float)
    covariance = np.asarray(
        model.cov_params(),
        dtype=float,
    )

    estimate = float(vector @ beta)
    variance = float(
        vector @ covariance @ vector
    )

    standard_error = math.sqrt(
        max(variance, 0.0)
    )

    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
    }


def diagnostic_status(
    rank,
    parameters,
    condition,
    maximum_leverage,
    high_leverage_count,
):
    if rank < parameters:
        return "FAIL_RANK_DEFICIENT"

    if maximum_leverage >= 0.99:
        return "REVIEW_SEVERE_LEVERAGE"

    if not math.isfinite(condition) or condition >= 1000:
        return "REVIEW_ILL_CONDITIONED"

    if high_leverage_count > 0:
        return "PASS_WITH_INFLUENTIAL_OBSERVATIONS"

    return "PASS"


result_rows = []
adjusted_rows = []
diagnostic_rows = []
descriptive_rows = []

contrast_sets = {
    "ADNI": [
        ("MCI_minus_CN", "MCI", "CN"),
        ("AD_minus_CN", "AD", "CN"),
        ("AD_minus_MCI", "AD", "MCI"),
    ],
    "NIFD": [
        ("bvFTD_minus_CN", "bvFTD", "CN"),
    ],
}

for analysis, dataset, original_work in analysis_specs:
    work = original_work.copy()

    expected = set(required_groups[dataset])
    observed = set(work["group"])

    if observed != expected:
        raise RuntimeError(
            f"{analysis} {dataset}: unexpected groups "
            f"{observed}; expected {expected}"
        )

    work["age_z"] = (
        work["age_model"]
        - work["age_model"].mean()
    ) / work["age_model"].std(ddof=0)

    work["eTIV_z"] = (
        work["eTIV_model"]
        - work["eTIV_model"].mean()
    ) / work["eTIV_model"].std(ddof=0)

    for outcome in all_features:
        family = (
            "PRIMARY"
            if outcome in primary_features
            else "SECONDARY"
        )

        for group in required_groups[dataset]:
            values = work.loc[
                work["group"] == group,
                outcome,
            ].to_numpy(dtype=float)

            descriptive_rows.append(
                {
                    "analysis": analysis,
                    "dataset": dataset,
                    "family": family,
                    "outcome": outcome,
                    "group": group,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "SD": float(
                        np.std(values, ddof=1)
                    ),
                    "median": float(
                        np.median(values)
                    ),
                    "Q1": float(
                        np.quantile(values, 0.25)
                    ),
                    "Q3": float(
                        np.quantile(values, 0.75)
                    ),
                }
            )

        covariates = (
            "age_z + C(sex_model) + C(site_model)"
        )

        if outcome in volume_features:
            covariates += " + eTIV_z"
            outcome_type = "VOLUME"
        else:
            outcome_type = "THICKNESS"

        raw_formula = (
            f"{outcome} ~ "
            "C(group, Treatment(reference='CN')) + "
            f"{covariates}"
        )

        raw_model = smf.ols(
            raw_formula,
            data=work,
        ).fit(cov_type="HC3")

        outcome_sd = work[outcome].std(ddof=0)

        if not math.isfinite(outcome_sd) or outcome_sd <= 0:
            raise RuntimeError(
                f"Invalid SD for {outcome}"
            )

        work["_standardized_outcome"] = (
            work[outcome] - work[outcome].mean()
        ) / outcome_sd

        standardized_formula = (
            "_standardized_outcome ~ "
            "C(group, Treatment(reference='CN')) + "
            f"{covariates}"
        )

        standardized_model = smf.ols(
            standardized_formula,
            data=work,
        ).fit(cov_type="HC3")

        design = np.asarray(
            raw_model.model.exog,
            dtype=float,
        )

        rank = int(
            np.linalg.matrix_rank(design)
        )

        singular = np.linalg.svd(
            design,
            compute_uv=False,
        )

        condition = (
            float("inf")
            if singular[-1] <= 1e-12
            else float(
                singular[0] / singular[-1]
            )
        )

        ordinary_model = smf.ols(
            raw_formula,
            data=work,
        ).fit()

        leverage = np.asarray(
            ordinary_model
            .get_influence()
            .hat_matrix_diag,
            dtype=float,
        )

        parameters = design.shape[1]
        leverage_threshold = (
            2.0 * parameters / len(work)
        )

        high_leverage_count = int(
            np.sum(leverage > leverage_threshold)
        )

        maximum_leverage = float(
            np.max(leverage)
        )

        status = diagnostic_status(
            rank,
            parameters,
            condition,
            maximum_leverage,
            high_leverage_count,
        )

        diagnostic_rows.append(
            {
                "analysis": analysis,
                "dataset": dataset,
                "family": family,
                "outcome": outcome,
                "outcome_type": outcome_type,
                "n": len(work),
                "parameters": parameters,
                "rank": rank,
                "condition_number": condition,
                "maximum_leverage": maximum_leverage,
                "leverage_threshold_2p_over_n": (
                    leverage_threshold
                ),
                "high_leverage_count": (
                    high_leverage_count
                ),
                "diagnostic_status": status,
            }
        )

        for group in required_groups[dataset]:
            estimate = marginal_group_estimate(
                raw_model,
                work,
                group,
            )

            adjusted_rows.append(
                {
                    "analysis": analysis,
                    "dataset": dataset,
                    "family": family,
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "group": group,
                    "n_model": len(work),
                    "adjusted_mean": estimate["estimate"],
                    "standard_error_HC3": (
                        estimate["standard_error"]
                    ),
                    "ci_low": estimate["ci_low"],
                    "ci_high": estimate["ci_high"],
                    "diagnostic_status": status,
                }
            )

        for (
            contrast,
            group_a,
            group_b,
        ) in contrast_sets[dataset]:
            raw_vector = vector_for_contrast(
                raw_model,
                group_a,
                group_b,
            )

            standardized_vector = vector_for_contrast(
                standardized_model,
                group_a,
                group_b,
            )

            raw_result = contrast_result(
                raw_model,
                raw_vector,
            )

            standardized_result = contrast_result(
                standardized_model,
                standardized_vector,
            )

            result_rows.append(
                {
                    "analysis": analysis,
                    "dataset": dataset,
                    "family": family,
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "contrast": contrast,
                    "group_A": group_a,
                    "group_B": group_b,
                    "n": len(work),
                    "raw_adjusted_difference": (
                        raw_result["estimate"]
                    ),
                    "raw_standard_error_HC3": (
                        raw_result["standard_error"]
                    ),
                    "raw_ci_low": (
                        raw_result["ci_low"]
                    ),
                    "raw_ci_high": (
                        raw_result["ci_high"]
                    ),
                    "standardized_adjusted_difference": (
                        standardized_result["estimate"]
                    ),
                    "standardized_standard_error_HC3": (
                        standardized_result[
                            "standard_error"
                        ]
                    ),
                    "standardized_ci_low": (
                        standardized_result["ci_low"]
                    ),
                    "standardized_ci_high": (
                        standardized_result["ci_high"]
                    ),
                    "p_HC3": (
                        standardized_result["p_HC3"]
                    ),
                    "diagnostic_status": status,
                }
            )

results = pd.DataFrame(result_rows)
adjusted = pd.DataFrame(adjusted_rows)
diagnostics = pd.DataFrame(diagnostic_rows)
descriptives = pd.DataFrame(descriptive_rows)

results["BH_q"] = np.nan

for (
    analysis,
    dataset,
    family,
), index in results.groupby(
    ["analysis", "dataset", "family"]
).groups.items():
    results.loc[index, "BH_q"] = bh_fdr(
        results.loc[index, "p_HC3"]
    )

results["significant_q05"] = np.where(
    results["BH_q"] < 0.05,
    "YES",
    "NO",
)

results.to_csv(
    table_dir / "FreeSurfer_atrophy_adjusted_results.tsv",
    sep="\t",
    index=False,
)

adjusted.to_csv(
    table_dir / "FreeSurfer_atrophy_adjusted_group_values.tsv",
    sep="\t",
    index=False,
)

diagnostics.to_csv(
    table_dir / "FreeSurfer_atrophy_model_diagnostics.tsv",
    sep="\t",
    index=False,
)

descriptives.to_csv(
    table_dir / "FreeSurfer_atrophy_raw_group_values.tsv",
    sep="\t",
    index=False,
)

# Forest plots for the prespecified primary family.
for dataset in ["ADNI", "NIFD"]:
    plot = results[
        (results["analysis"] == "PRIMARY_SHARED_SITES")
        & (results["dataset"] == dataset)
        & (results["family"] == "PRIMARY")
    ].copy()

    plot = plot.sort_values(
        ["contrast", "outcome"],
        ascending=[True, True],
    )

    labels = [
        f"{row.contrast} | "
        f"{row.outcome.replace('_', ' ')}"
        for row in plot.itertuples()
    ]

    y = np.arange(len(plot))
    estimate = plot[
        "standardized_adjusted_difference"
    ].to_numpy()

    low = plot[
        "standardized_ci_low"
    ].to_numpy()

    high = plot[
        "standardized_ci_high"
    ].to_numpy()

    colors = [
        "#b2182b" if q < 0.05 else "#2166ac"
        for q in plot["BH_q"]
    ]

    height = max(6.0, 0.38 * len(plot) + 1.8)

    fig, ax = plt.subplots(
        figsize=(10, height)
    )

    for position, effect, lower, upper, color in zip(
        y,
        estimate,
        low,
        high,
        colors,
    ):
        ax.errorbar(
            effect,
            position,
            xerr=[
                [effect - lower],
                [upper - effect],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Adjusted group difference (outcome SD units)"
    )
    ax.set_title(
        f"{dataset} FreeSurfer primary atrophy results\n"
        "Red: BH-FDR q < 0.05"
    )
    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    stem = (
        f"{dataset}_FreeSurfer_primary_atrophy_forest"
    )

    fig.savefig(
        png_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        svg_dir / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)

# Adjusted group-value figures for primary outcomes.
for dataset in ["ADNI", "NIFD"]:
    plot = adjusted[
        (adjusted["analysis"] == "PRIMARY_SHARED_SITES")
        & (adjusted["dataset"] == dataset)
        & (adjusted["family"] == "PRIMARY")
    ].copy()

    groups = required_groups[dataset]
    colors = {
        "CN": "#4C78A8",
        "MCI": "#F2CF5B",
        "AD": "#E45756",
        "bvFTD": "#B279A2",
    }

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13, 11),
    )

    axes = axes.flatten()

    for axis, outcome in zip(
        axes,
        primary_features,
    ):
        values = plot[
            plot["outcome"] == outcome
        ].set_index("group")

        positions = np.arange(len(groups))

        means = np.array(
            [
                values.loc[group, "adjusted_mean"]
                for group in groups
            ]
        )

        lower = np.array(
            [
                values.loc[group, "ci_low"]
                for group in groups
            ]
        )

        upper = np.array(
            [
                values.loc[group, "ci_high"]
                for group in groups
            ]
        )

        axis.errorbar(
            positions,
            means,
            yerr=[
                means - lower,
                upper - means,
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.2,
        )

        axis.scatter(
            positions,
            means,
            s=55,
            c=[colors[group] for group in groups],
            zorder=3,
        )

        axis.set_xticks(positions)
        axis.set_xticklabels(groups)
        axis.set_title(
            outcome.replace("_", " "),
            fontsize=9,
        )
        axis.grid(
            axis="y",
            alpha=0.2,
        )

    for axis in axes[len(primary_features):]:
        axis.axis("off")

    fig.suptitle(
        f"{dataset}: adjusted FreeSurfer group values\n"
        "95% HC3 confidence intervals",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    stem = (
        f"{dataset}_FreeSurfer_adjusted_group_values"
    )

    fig.savefig(
        png_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        svg_dir / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)

primary_results = results[
    (results["analysis"] == "PRIMARY_SHARED_SITES")
    & (results["family"] == "PRIMARY")
].copy()

summary = [
    "FREESURFER ADJUSTED ATROPHY ANALYSIS",
    "",
    f"Input: {input_path}",
    f"Age column: {age_field}",
    f"Sex column: {sex_field}",
    f"Site column: {site_field}",
    "",
    "PRIMARY DESIGN",
    (
        "Sites containing all relevant diagnostic groups "
        "within each dataset."
    ),
    (
        "Volumes: diagnosis + age + sex + site + eTIV."
    ),
    (
        "Thickness: diagnosis + age + sex + site."
    ),
    "Inference: HC3 robust standard errors.",
    (
        "BH-FDR correction is performed separately within "
        "analysis, dataset and feature family."
    ),
    "",
    "PRIMARY COHORTS",
]

for row in cohort_summary_rows:
    if row["analysis"] != "PRIMARY_SHARED_SITES":
        continue

    summary.append(
        f'{row["dataset"]}: n={row["n_total"]}; '
        f'{row["group_counts"]}; '
        f'sites={row["n_sites"]}'
    )

summary.extend(
    [
        "",
        "PRIMARY RESULTS WITH BH-FDR q < 0.05",
    ]
)

significant = primary_results[
    primary_results["BH_q"] < 0.05
].sort_values(
    ["dataset", "BH_q", "outcome", "contrast"]
)

if significant.empty:
    summary.append("NONE")
else:
    for row in significant.itertuples():
        summary.append(
            f"{row.dataset} | {row.outcome} | "
            f"{row.contrast} | "
            f"standardized difference="
            f"{row.standardized_adjusted_difference:.4f} | "
            f"95% CI=[{row.standardized_ci_low:.4f}, "
            f"{row.standardized_ci_high:.4f}] | "
            f"p={row.p_HC3:.6g} | "
            f"q={row.BH_q:.6g}"
        )

summary.extend(
    [
        "",
        "INTERPRETATION",
        (
            "For brain volumes and cortical thickness, a "
            "negative patient-minus-control effect indicates "
            "smaller volume or thinner cortex."
        ),
        (
            "For lateral ventricular volume, a positive "
            "patient-minus-control effect indicates greater "
            "ventricular enlargement."
        ),
        (
            "ADNI and NIFD were analysed separately and were "
            "not pooled into one regression."
        ),
        (
            "Sensitivity full-cohort models retain subjects "
            "from diagnostically unsupported sites and must "
            "be interpreted with their leverage diagnostics."
        ),
        (
            "Statistical results remain provisional pending "
            "final anatomical FreeSurfer segmentation and "
            "surface QC."
        ),
    ]
)

(out / "FreeSurfer_atrophy_analysis_summary.txt").write_text(
    "\n".join(summary) + "\n",
    encoding="utf-8",
)

versions = [
    f"Python\t{platform.python_version()}",
    f"NumPy\t{np.__version__}",
    f"pandas\t{pd.__version__}",
    f"SciPy\t{scipy.__version__}",
    f"statsmodels\t{statsmodels.__version__}",
    f"matplotlib\t{matplotlib.__version__}",
]

(out / "software_versions.txt").write_text(
    "\n".join(versions) + "\n",
    encoding="utf-8",
)

print("===== PRIMARY COHORTS =====")

for row in cohort_summary_rows:
    if row["analysis"] == "PRIMARY_SHARED_SITES":
        print(
            row["dataset"],
            row["n_total"],
            row["group_counts"],
            f'sites={row["n_sites"]}',
            sep="\t",
        )

print(
    "\n===== PRIMARY BH-FDR SIGNIFICANT RESULTS ====="
)

if significant.empty:
    print("NONE")
else:
    print(
        significant[
            [
                "dataset",
                "outcome",
                "contrast",
                "raw_adjusted_difference",
                "standardized_adjusted_difference",
                "standardized_ci_low",
                "standardized_ci_high",
                "p_HC3",
                "BH_q",
                "diagnostic_status",
            ]
        ].to_csv(
            sep="\t",
            index=False,
        )
    )

print("\nOutput:", out)
