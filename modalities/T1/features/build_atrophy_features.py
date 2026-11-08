import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

features_path = Path(os.environ["FEATURES"])
cohort_path = Path(os.environ["COHORT"])
out = Path(os.environ["OUT"])
tables = out / "tables"

features = pd.read_csv(
    features_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

cohort = pd.read_csv(
    cohort_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)


def select(columns, candidates, description):
    lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise RuntimeError(
        f"Could not identify {description}. "
        f"Available columns: {list(columns)}"
    )


cohort_subject = select(
    cohort.columns,
    ["subject", "subject_id"],
    "cohort subject",
)

age_field = select(
    cohort.columns,
    ["age_at_DWI_years", "age", "age_years"],
    "age",
)

sex_field = select(
    cohort.columns,
    ["sex", "gender"],
    "sex",
)

site_field = select(
    cohort.columns,
    ["acquisition_site", "site", "site_id"],
    "site",
)

cohort = cohort.rename(
    columns={
        cohort_subject: "subject",
        age_field: "age",
        sex_field: "sex",
        site_field: "site",
    }
)

for frame in [features, cohort]:
    for field in ["dataset", "group", "subject"]:
        frame[field] = (
            frame[field]
            .astype(str)
            .str.strip()
        )

if features.duplicated(["dataset", "subject"]).any():
    raise RuntimeError(
        "Duplicate subjects in FreeSurfer feature table"
    )

if cohort.duplicated(["dataset", "subject"]).any():
    raise RuntimeError(
        "Duplicate subjects in cohort table"
    )

merged = cohort[
    [
        "dataset",
        "group",
        "subject",
        "age",
        "sex",
        "site",
    ]
].merge(
    features.drop(columns=["group"]),
    on=["dataset", "subject"],
    how="inner",
    validate="one_to_one",
)

if len(merged) != 343:
    raise RuntimeError(
        f"Expected 343 merged records; found {len(merged)}"
    )

merged["age"] = pd.to_numeric(
    merged["age"],
    errors="coerce",
)

numeric_columns = [
    column
    for column in merged.columns
    if (
        column.endswith("_mm3")
        or column.endswith("_mm")
        or column.endswith("_mm2")
    )
]

for column in numeric_columns:
    merged[column] = pd.to_numeric(
        merged[column],
        errors="coerce",
    )


def weighted_thickness(frame, regions):
    numerator = np.zeros(len(frame), dtype=float)
    denominator = np.zeros(len(frame), dtype=float)
    valid = np.ones(len(frame), dtype=bool)

    for region in regions:
        thickness_column = (
            f"bilateral_{region}_thickness_mm"
        )
        area_column = (
            f"bilateral_{region}_area_mm2"
        )

        if thickness_column not in frame.columns:
            raise RuntimeError(
                f"Missing column: {thickness_column}"
            )

        if area_column not in frame.columns:
            raise RuntimeError(
                f"Missing column: {area_column}"
            )

        thickness = frame[
            thickness_column
        ].to_numpy(dtype=float)

        area = frame[
            area_column
        ].to_numpy(dtype=float)

        current_valid = (
            np.isfinite(thickness)
            & np.isfinite(area)
            & (area > 0)
        )

        valid &= current_valid
        numerator += np.where(
            current_valid,
            thickness * area,
            0.0,
        )
        denominator += np.where(
            current_valid,
            area,
            0.0,
        )

    result = np.full(len(frame), np.nan)

    final_valid = (
        valid
        & np.isfinite(numerator)
        & (denominator > 0)
    )

    result[final_valid] = (
        numerator[final_valid]
        / denominator[final_valid]
    )

    return result


frontal_regions = [
    "superiorfrontal",
    "rostralmiddlefrontal",
    "caudalmiddlefrontal",
    "lateralorbitofrontal",
    "medialorbitofrontal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "frontalpole",
]

temporal_regions = [
    "superiortemporal",
    "middletemporal",
    "inferiortemporal",
    "temporalpole",
    "fusiform",
    "entorhinal",
    "parahippocampal",
]

medial_temporal_regions = [
    "entorhinal",
    "parahippocampal",
]

anterior_cingulate_regions = [
    "rostralanteriorcingulate",
    "caudalanteriorcingulate",
]

orbitofrontal_regions = [
    "lateralorbitofrontal",
    "medialorbitofrontal",
]

merged["frontal_composite_thickness_mm"] = (
    weighted_thickness(
        merged,
        frontal_regions,
    )
)

merged["temporal_composite_thickness_mm"] = (
    weighted_thickness(
        merged,
        temporal_regions,
    )
)

merged[
    "medial_temporal_composite_thickness_mm"
] = weighted_thickness(
    merged,
    medial_temporal_regions,
)

merged[
    "anterior_cingulate_composite_thickness_mm"
] = weighted_thickness(
    merged,
    anterior_cingulate_regions,
)

merged[
    "orbitofrontal_composite_thickness_mm"
] = weighted_thickness(
    merged,
    orbitofrontal_regions,
)

merged["insula_thickness_mm"] = merged[
    "bilateral_insula_thickness_mm"
]

# Convert selected volumes from mm3 to mL.
volume_mapping = {
    "eTIV_ml": "eTIV_mm3",
    "cortical_gray_volume_ml": "CortexVol_mm3",
    "cerebral_white_matter_volume_ml": (
        "CerebralWhiteMatterVol_mm3"
    ),
    "subcortical_gray_volume_ml": (
        "SubCortGrayVol_mm3"
    ),
    "hippocampus_volume_ml": (
        "bilateral_hippocampus_volume_mm3"
    ),
    "amygdala_volume_ml": (
        "bilateral_amygdala_volume_mm3"
    ),
    "lateral_ventricle_volume_ml": (
        "bilateral_lateral_ventricle_volume_mm3"
    ),
    "corpus_callosum_volume_ml": (
        "corpus_callosum_volume_mm3"
    ),
}

for output_column, source_column in (
    volume_mapping.items()
):
    if source_column not in merged.columns:
        raise RuntimeError(
            f"Missing source volume: {source_column}"
        )

    merged[output_column] = (
        merged[source_column] / 1000.0
    )

analysis_columns = [
    "eTIV_ml",
    "cortical_gray_volume_ml",
    "cerebral_white_matter_volume_ml",
    "subcortical_gray_volume_ml",
    "hippocampus_volume_ml",
    "amygdala_volume_ml",
    "lateral_ventricle_volume_ml",
    "corpus_callosum_volume_ml",
    "bilateral_MeanThickness_mm",
    "frontal_composite_thickness_mm",
    "temporal_composite_thickness_mm",
    "medial_temporal_composite_thickness_mm",
    "anterior_cingulate_composite_thickness_mm",
    "orbitofrontal_composite_thickness_mm",
    "insula_thickness_mm",
]

audit_rows = []

for column in analysis_columns:
    values = merged[column]

    finite = np.isfinite(values.to_numpy(dtype=float))

    audit_rows.append(
        {
            "feature": column,
            "n_total": len(values),
            "n_finite": int(finite.sum()),
            "n_missing": int((~finite).sum()),
            "minimum": float(
                np.nanmin(values)
            ),
            "Q1": float(
                np.nanquantile(values, 0.25)
            ),
            "median": float(
                np.nanmedian(values)
            ),
            "Q3": float(
                np.nanquantile(values, 0.75)
            ),
            "maximum": float(
                np.nanmax(values)
            ),
        }
    )

group_rows = []

for (dataset, group), rows in merged.groupby(
    ["dataset", "group"]
):
    for column in analysis_columns:
        values = rows[column].to_numpy(dtype=float)

        group_rows.append(
            {
                "dataset": dataset,
                "group": group,
                "feature": column,
                "n": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)),
                "SD": float(
                    np.nanstd(values, ddof=1)
                ),
                "median": float(np.nanmedian(values)),
                "Q1": float(
                    np.nanquantile(values, 0.25)
                ),
                "Q3": float(
                    np.nanquantile(values, 0.75)
                ),
            }
        )

analysis_table = merged[
    [
        "dataset",
        "group",
        "subject",
        "age",
        "sex",
        "site",
        "freesurfer_version",
        *analysis_columns,
    ]
].copy()

analysis_path = (
    tables / "FreeSurfer_atrophy_analysis_table.tsv"
)

audit_path = (
    tables / "FreeSurfer_atrophy_feature_audit.tsv"
)

group_path = (
    tables / "FreeSurfer_atrophy_raw_group_values.tsv"
)

analysis_table.to_csv(
    analysis_path,
    sep="\t",
    index=False,
    na_rep="NA",
)

pd.DataFrame(audit_rows).to_csv(
    audit_path,
    sep="\t",
    index=False,
    na_rep="NA",
)

pd.DataFrame(group_rows).to_csv(
    group_path,
    sep="\t",
    index=False,
    na_rep="NA",
)

missing_features = [
    row
    for row in audit_rows
    if row["n_missing"] > 0
]

counts = (
    analysis_table.groupby(["dataset", "group"])
    .size()
    .to_dict()
)

summary = [
    "FREESURFER ATROPHY FEATURE PREFLIGHT",
    "",
    f"Subjects: {len(analysis_table)}",
    f"Features: {len(analysis_columns)}",
    f"Features with missing values: {len(missing_features)}",
    "",
    "COUNTS",
]

for key, count in sorted(counts.items()):
    dataset, group = key
    summary.append(
        f"{dataset}\t{group}\t{count}"
    )

summary.extend(
    [
        "",
        "PRIMARY ATROPHY FAMILY",
        "hippocampus_volume_ml",
        "medial_temporal_composite_thickness_mm",
        "frontal_composite_thickness_mm",
        "temporal_composite_thickness_mm",
        "anterior_cingulate_composite_thickness_mm",
        "insula_thickness_mm",
        "lateral_ventricle_volume_ml",
        "",
        "SECONDARY FAMILY",
        "cortical_gray_volume_ml",
        "cerebral_white_matter_volume_ml",
        "subcortical_gray_volume_ml",
        "amygdala_volume_ml",
        "corpus_callosum_volume_ml",
        "bilateral_MeanThickness_mm",
        "orbitofrontal_composite_thickness_mm",
        "",
        "MODEL RULE",
        (
            "Volumes: outcome ~ diagnosis + age + sex + "
            "site + eTIV."
        ),
        (
            "Thickness: outcome ~ diagnosis + age + sex + site."
        ),
        (
            "Do not divide cortical thickness by eTIV."
        ),
    ]
)

summary_path = (
    out / "FreeSurfer_atrophy_preflight_summary.txt"
)

summary_path.write_text(
    "\n".join(summary) + "\n",
    encoding="utf-8",
)

checksum_path = out / "SHA256SUMS.txt"

files = [
    analysis_path,
    audit_path,
    group_path,
    summary_path,
]

with checksum_path.open(
    "w",
    encoding="utf-8",
) as handle:
    for path in files:
        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        handle.write(
            f"{digest}  {path.relative_to(out)}\n"
        )

print("\n".join(summary))

if missing_features:
    print("\nFEATURES WITH MISSING VALUES")

    for row in missing_features:
        print(
            f"{row['feature']}\t"
            f"missing={row['n_missing']}"
        )

    raise RuntimeError(
        "Atrophy analysis features contain missing values"
    )
