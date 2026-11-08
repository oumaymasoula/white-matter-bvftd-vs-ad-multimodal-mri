import csv
import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np


coarse_4mm_subject = os.environ["WM_COARSE_4MM_SUBJECT"]
coarse_5mm_subject = os.environ["WM_COARSE_5MM_SUBJECT"]
tbssroot = Path(os.environ["TBSSROOT"])
stats = Path(os.environ["STATS"])
order_file = Path(os.environ["ORDER"])
coverage_file = Path(os.environ["COVERAGE"])
negative_file = Path(os.environ["NEGATIVE"])
atlas_file = Path(os.environ["ATLAS"])
skeleton_file = Path(os.environ["SKELETON"])
extract = Path(os.environ["EXTRACT"])

output_file = extract / "jhu_subject_tract_FA_MD_AD_RD.tsv"
summary_file = extract / "jhu_tract_extraction_summary.txt"
status_file = extract / "jhu_tract_extraction_status.txt"
failure_file = extract / "jhu_tract_extraction_failures.tsv"
tract_file = extract / "jhu_tract_inventory.tsv"

metrics = ("FA", "MD", "AD", "RD")
atlas_threshold = 25.0
expected_subjects = 347
expected_tracts = 20
expected_cells = expected_subjects * expected_tracts


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def integer(value, label):
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid integer for {label}: {value!r}"
        ) from exc


def number(value, label):
    try:
        result = float(value)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid number for {label}: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise RuntimeError(
            f"Nonfinite number for {label}: {value!r}"
        )

    return result


subjects = read_tsv(order_file)
coverage_rows = read_tsv(coverage_file)
negative_rows = read_tsv(negative_file)

if len(subjects) != expected_subjects:
    raise RuntimeError(
        f"Expected 347 subjects; found {len(subjects)}"
    )

if len(coverage_rows) != expected_cells:
    raise RuntimeError(
        f"Expected 6940 coverage rows; found {len(coverage_rows)}"
    )

if len(negative_rows) != expected_cells:
    raise RuntimeError(
        f"Expected 6940 negative-audit rows; found {len(negative_rows)}"
    )

subjects.sort(key=lambda row: integer(row["volume_index"], "volume_index"))

for expected_volume, subject in enumerate(subjects):
    observed_volume = integer(subject["volume_index"], "volume_index")

    if observed_volume != expected_volume:
        raise RuntimeError(
            "Subject-order table is not contiguous: "
            f"expected {expected_volume}, found {observed_volume}"
        )

    expected_task = expected_volume + 1
    observed_task = integer(subject["task"], "task")

    if observed_task != expected_task:
        raise RuntimeError(
            "Task/order mismatch: "
            f"expected task {expected_task}, found {observed_task}"
        )


coverage_lookup = {}
tract_names = {}

for row in coverage_rows:
    task = integer(row["task"], "coverage task")
    tract_index = integer(
        row["tract_volume_index"],
        "coverage tract_volume_index",
    )
    key = (task, tract_index)

    if key in coverage_lookup:
        raise RuntimeError(f"Duplicate coverage key: {key}")

    coverage_lookup[key] = row

    tract_name = row["tract"]
    previous = tract_names.get(tract_index)

    if previous is not None and previous != tract_name:
        raise RuntimeError(
            f"Inconsistent tract name for index {tract_index}"
        )

    tract_names[tract_index] = tract_name


negative_lookup = {}

for row in negative_rows:
    task = integer(row["task"], "negative task")
    tract_index = integer(row["tract_index"], "negative tract_index")
    key = (task, tract_index)

    if key in negative_lookup:
        raise RuntimeError(f"Duplicate negative-audit key: {key}")

    negative_lookup[key] = row


expected_keys = {
    (task, tract)
    for task in range(1, expected_subjects + 1)
    for tract in range(expected_tracts)
}

if set(coverage_lookup) != expected_keys:
    missing = sorted(expected_keys - set(coverage_lookup))
    extra = sorted(set(coverage_lookup) - expected_keys)
    raise RuntimeError(
        f"Coverage key mismatch; missing={missing[:5]}, extra={extra[:5]}"
    )

if set(negative_lookup) != expected_keys:
    missing = sorted(expected_keys - set(negative_lookup))
    extra = sorted(set(negative_lookup) - expected_keys)
    raise RuntimeError(
        f"Negative key mismatch; missing={missing[:5]}, extra={extra[:5]}"
    )

if set(tract_names) != set(range(expected_tracts)):
    raise RuntimeError(
        f"Expected tract indices 0–19; found {sorted(tract_names)}"
    )


skeleton_img = nib.load(skeleton_file)
atlas_img = nib.load(atlas_file)

skeleton = np.asarray(skeleton_img.dataobj) > 0
atlas_shape = atlas_img.shape

if len(atlas_shape) != 4 or atlas_shape[3] != expected_tracts:
    raise RuntimeError(
        f"Unexpected atlas dimensions: {atlas_shape}"
    )

if atlas_shape[:3] != skeleton.shape:
    raise RuntimeError(
        "Atlas/skeleton geometry mismatch: "
        f"{atlas_shape[:3]} versus {skeleton.shape}"
    )

if not np.allclose(
    atlas_img.affine,
    skeleton_img.affine,
    rtol=0,
    atol=1e-4,
):
    raise RuntimeError("Atlas and skeleton affine matrices differ")


metric_images = {}

for metric in metrics:
    path = stats / f"all_{metric}_skeletonised.nii.gz"
    image = nib.load(path)

    if image.shape[:3] != skeleton.shape:
        raise RuntimeError(
            f"{metric} spatial geometry mismatch: {image.shape}"
        )

    if len(image.shape) != 4 or image.shape[3] != expected_subjects:
        raise RuntimeError(
            f"{metric} expected 347 volumes; found {image.shape}"
        )

    if not np.allclose(
        image.affine,
        skeleton_img.affine,
        rtol=0,
        atol=1e-4,
    ):
        raise RuntimeError(f"{metric} affine differs from skeleton")

    metric_images[metric] = image


tract_masks = {}
tract_weights = {}
tract_inventory = []

for tract_index in range(expected_tracts):
    probability = np.asarray(
        atlas_img.dataobj[..., tract_index],
        dtype=np.float64,
    )

    if not np.isfinite(probability).all():
        raise RuntimeError(
            f"Atlas tract {tract_index} contains nonfinite values"
        )

    mask = skeleton & (probability >= atlas_threshold)
    weights = probability[mask]
    voxels = int(mask.sum())

    if voxels < 100:
        raise RuntimeError(
            f"Tract {tract_index} contains only {voxels} skeleton voxels"
        )

    if weights.size != voxels or np.sum(weights) <= 0:
        raise RuntimeError(
            f"Invalid weights for tract {tract_index}"
        )

    tract_masks[tract_index] = mask
    tract_weights[tract_index] = weights

    tract_inventory.append({
        "tract_index": tract_index,
        "tract": tract_names[tract_index],
        "atlas_probability_threshold_percent": atlas_threshold,
        "skeleton_voxels": voxels,
        "probability_sum": float(np.sum(weights)),
        "probability_minimum": float(np.min(weights)),
        "probability_maximum": float(np.max(weights)),
    })


with tract_file.open("w", newline="") as handle:
    fields = list(tract_inventory[0])
    writer = csv.DictWriter(handle, fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(tract_inventory)


output_fields = [
    "task",
    "volume_index_zero_based",
    "dataset",
    "group",
    "subject",
    "adni_site_prefix",
    "tract_index_zero_based",
    "tract",
    "atlas_probability_threshold_percent",
    "tract_skeleton_voxels",
    "supported_tract_voxels",
    "unsupported_tract_voxels",
    "unweighted_support_percent",
    "probability_weighted_support_percent",
    "primary_coverage_eligible_ge95",
    "strict_coverage_eligible_ge99",
    "coarse_resolution_4mm_flag",
    "coarse_resolution_5mm_flag",
    "coarse_resolution_sensitivity_eligible",
]

for metric in metrics:
    output_fields.extend([
        f"{metric}_finite_voxels",
        f"{metric}_minimum",
        f"{metric}_maximum",
        f"{metric}_mean",
        f"{metric}_median",
        f"{metric}_standard_deviation",
        f"{metric}_q25",
        f"{metric}_q75",
        f"{metric}_iqr",
        f"{metric}_probability_weighted_mean",
        f"{metric}_negative_voxels",
        f"{metric}_negative_percent",
    ])

output_fields.extend([
    "any_negative_voxels",
    "any_negative_percent",
    "negative_classification",
    "primary_negative_policy_eligible",
    "sensitivity_negative_lt5_percent_eligible",
    "strict_negative_lt1_percent_eligible",
    "primary_analysis_eligible",
    "strict_coverage_sensitivity_eligible",
    "negative_5percent_sensitivity_eligible",
    "negative_1percent_sensitivity_eligible",
    "coarse_resolution_exclusion_sensitivity_eligible",
])


failures = []
rows_written = 0
primary_failures = 0
strict_coverage_failures = 0
negative_5_failures = 0
negative_1_failures = 0
coarse_failures = 0
identity_max_mean_error = 0.0
identity_max_weighted_error = 0.0
audit_max_mean_difference = 0.0
audit_max_median_difference = 0.0


with output_file.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=output_fields,
        delimiter="\t",
    )
    writer.writeheader()

    for subject in subjects:
        task = integer(subject["task"], "task")
        volume_index = integer(
            subject["volume_index"],
            "volume_index",
        )
        dataset = subject["dataset"]
        group = subject["group"]
        subject_id = subject["subject"]

        if dataset == "ADNI" and "S" in subject_id:
            adni_site = subject_id.replace("sub-", "", 1).split("S", 1)[0]
        else:
            adni_site = ""

        subject_volumes = {
            metric: np.asarray(
                metric_images[metric].dataobj[..., volume_index],
                dtype=np.float64,
            )
            for metric in metrics
        }

        for metric, volume in subject_volumes.items():
            if not np.isfinite(volume[skeleton]).all():
                failures.append({
                    "task": task,
                    "dataset": dataset,
                    "group": group,
                    "subject": subject_id,
                    "tract_index": "",
                    "failure": f"{metric}_NONFINITE_SKELETON_VALUE",
                })

        for tract_index in range(expected_tracts):
            key = (task, tract_index)
            coverage = coverage_lookup[key]
            negative = negative_lookup[key]
            tract_name = tract_names[tract_index]
            mask = tract_masks[tract_index]
            weights = tract_weights[tract_index]
            roi_voxels = int(mask.sum())

            identity_fields = (
                ("dataset", dataset),
                ("group", group),
                ("subject", subject_id),
            )

            for field, expected in identity_fields:
                if coverage[field] != expected:
                    raise RuntimeError(
                        f"Coverage identity mismatch at {key}: {field}"
                    )

                if negative[field] != expected:
                    raise RuntimeError(
                        f"Negative identity mismatch at {key}: {field}"
                    )

            if coverage["tract"] != tract_name:
                raise RuntimeError(
                    f"Coverage tract-name mismatch at {key}"
                )

            if negative["tract"] != tract_name:
                raise RuntimeError(
                    f"Negative tract-name mismatch at {key}"
                )

            coverage_roi = integer(
                coverage["tract_skeleton_voxels"],
                "coverage tract_skeleton_voxels",
            )
            negative_roi = integer(
                negative["tract_skeleton_voxels"],
                "negative tract_skeleton_voxels",
            )

            if coverage_roi != roi_voxels or negative_roi != roi_voxels:
                raise RuntimeError(
                    f"ROI-size mismatch at {key}: "
                    f"extracted={roi_voxels}, "
                    f"coverage={coverage_roi}, "
                    f"negative={negative_roi}"
                )

            support_unweighted = number(
                coverage["unweighted_support_percent"],
                "unweighted support",
            )
            support_weighted = number(
                coverage["probability_weighted_support_percent"],
                "weighted support",
            )

            primary_coverage = (
                roi_voxels >= 100
                and support_unweighted >= 95.0
                and support_weighted >= 95.0
            )
            strict_coverage = (
                roi_voxels >= 100
                and support_unweighted >= 99.0
                and support_weighted >= 99.0
            )

            coarse_4mm = (
                task == 242
                and dataset == "ADNI"
                and group == "MCI"
                and subject_id == coarse_4mm_subject
            )
            coarse_5mm = (
                task == 327
                and dataset == "ADNI"
                and group == "AD"
                and subject_id == coarse_5mm_subject
            )
            coarse_eligible = not (coarse_4mm or coarse_5mm)

            output = {
                "task": task,
                "volume_index_zero_based": volume_index,
                "dataset": dataset,
                "group": group,
                "subject": subject_id,
                "adni_site_prefix": adni_site,
                "tract_index_zero_based": tract_index,
                "tract": tract_name,
                "atlas_probability_threshold_percent": atlas_threshold,
                "tract_skeleton_voxels": roi_voxels,
                "supported_tract_voxels": integer(
                    coverage["supported_tract_voxels"],
                    "supported tract voxels",
                ),
                "unsupported_tract_voxels": integer(
                    coverage["unsupported_tract_voxels"],
                    "unsupported tract voxels",
                ),
                "unweighted_support_percent": support_unweighted,
                "probability_weighted_support_percent": support_weighted,
                "primary_coverage_eligible_ge95": (
                    "YES" if primary_coverage else "NO"
                ),
                "strict_coverage_eligible_ge99": (
                    "YES" if strict_coverage else "NO"
                ),
                "coarse_resolution_4mm_flag": (
                    "YES" if coarse_4mm else "NO"
                ),
                "coarse_resolution_5mm_flag": (
                    "YES" if coarse_5mm else "NO"
                ),
                "coarse_resolution_sensitivity_eligible": (
                    "YES" if coarse_eligible else "NO"
                ),
            }

            extracted = {}

            for metric in metrics:
                values = subject_volumes[metric][mask]

                if values.size != roi_voxels:
                    raise RuntimeError(
                        f"{metric} ROI-size mismatch at {key}"
                    )

                finite = np.isfinite(values)
                finite_values = values[finite]
                finite_weights = weights[finite]

                if finite_values.size != roi_voxels:
                    failures.append({
                        "task": task,
                        "dataset": dataset,
                        "group": group,
                        "subject": subject_id,
                        "tract_index": tract_index,
                        "failure": f"{metric}_NONFINITE_TRACT_VALUE",
                    })
                    continue

                q25, median, q75 = np.percentile(
                    finite_values,
                    [25, 50, 75],
                )
                mean = float(np.mean(finite_values))
                weighted_mean = float(
                    np.average(finite_values, weights=finite_weights)
                )
                negative_voxels = int(np.count_nonzero(
                    finite_values < -1e-8
                ))
                negative_percent = (
                    100.0 * negative_voxels / finite_values.size
                )

                extracted[metric] = {
                    "mean": mean,
                    "median": float(median),
                    "weighted_mean": weighted_mean,
                    "negative_voxels": negative_voxels,
                    "negative_percent": negative_percent,
                }

                output.update({
                    f"{metric}_finite_voxels": int(finite_values.size),
                    f"{metric}_minimum": float(np.min(finite_values)),
                    f"{metric}_maximum": float(np.max(finite_values)),
                    f"{metric}_mean": mean,
                    f"{metric}_median": float(median),
                    f"{metric}_standard_deviation": float(
                        np.std(finite_values, ddof=1)
                    ),
                    f"{metric}_q25": float(q25),
                    f"{metric}_q75": float(q75),
                    f"{metric}_iqr": float(q75 - q25),
                    f"{metric}_probability_weighted_mean": weighted_mean,
                    f"{metric}_negative_voxels": negative_voxels,
                    f"{metric}_negative_percent": negative_percent,
                })

                if metric in ("MD", "AD", "RD"):
                    audit_count = integer(
                        negative[f"{metric}_negative_voxels"],
                        f"audit {metric} negative count",
                    )
                    audit_percent = number(
                        negative[f"{metric}_negative_percent"],
                        f"audit {metric} negative percent",
                    )
                    audit_mean = number(
                        negative[f"{metric}_mean_all_values"],
                        f"audit {metric} mean",
                    )
                    audit_median = number(
                        negative[f"{metric}_median_all_values"],
                        f"audit {metric} median",
                    )

                    if audit_count != negative_voxels:
                        raise RuntimeError(
                            f"{metric} negative-count mismatch at {key}"
                        )

                    if abs(audit_percent - negative_percent) > 1e-8:
                        raise RuntimeError(
                            f"{metric} negative-percent mismatch at {key}"
                        )

                    audit_max_mean_difference = max(
                        audit_max_mean_difference,
                        abs(audit_mean - mean),
                    )
                    audit_max_median_difference = max(
                        audit_max_median_difference,
                        abs(audit_median - float(median)),
                    )

            if len(extracted) != len(metrics):
                continue

            mean_identity = abs(
                extracted["MD"]["mean"]
                - (
                    extracted["AD"]["mean"]
                    + 2.0 * extracted["RD"]["mean"]
                ) / 3.0
            )
            weighted_identity = abs(
                extracted["MD"]["weighted_mean"]
                - (
                    extracted["AD"]["weighted_mean"]
                    + 2.0 * extracted["RD"]["weighted_mean"]
                ) / 3.0
            )

            identity_max_mean_error = max(
                identity_max_mean_error,
                mean_identity,
            )
            identity_max_weighted_error = max(
                identity_max_weighted_error,
                weighted_identity,
            )

            any_negative_voxels = integer(
                negative["any_negative_voxels"],
                "any_negative_voxels",
            )
            any_negative_percent = number(
                negative["any_negative_percent"],
                "any_negative_percent",
            )

            sensitivity_lt5 = any_negative_percent < 5.0
            sensitivity_lt1 = any_negative_percent < 1.0
            primary_analysis = primary_coverage

            output.update({
                "any_negative_voxels": any_negative_voxels,
                "any_negative_percent": any_negative_percent,
                "negative_classification": negative["classification"],
                "primary_negative_policy_eligible": "YES",
                "sensitivity_negative_lt5_percent_eligible": (
                    "YES" if sensitivity_lt5 else "NO"
                ),
                "strict_negative_lt1_percent_eligible": (
                    "YES" if sensitivity_lt1 else "NO"
                ),
                "primary_analysis_eligible": (
                    "YES" if primary_analysis else "NO"
                ),
                "strict_coverage_sensitivity_eligible": (
                    "YES" if strict_coverage else "NO"
                ),
                "negative_5percent_sensitivity_eligible": (
                    "YES"
                    if primary_coverage and sensitivity_lt5
                    else "NO"
                ),
                "negative_1percent_sensitivity_eligible": (
                    "YES"
                    if primary_coverage and sensitivity_lt1
                    else "NO"
                ),
                "coarse_resolution_exclusion_sensitivity_eligible": (
                    "YES"
                    if primary_coverage and coarse_eligible
                    else "NO"
                ),
            })

            writer.writerow(output)
            rows_written += 1

            if not primary_analysis:
                primary_failures += 1
            if not strict_coverage:
                strict_coverage_failures += 1
            if not sensitivity_lt5:
                negative_5_failures += 1
            if not sensitivity_lt1:
                negative_1_failures += 1
            if not coarse_eligible:
                coarse_failures += 1


with failure_file.open("w", newline="") as handle:
    fields = [
        "task",
        "dataset",
        "group",
        "subject",
        "tract_index",
        "failure",
    ]
    writer = csv.DictWriter(handle, fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(failures)


if rows_written != expected_cells:
    raise RuntimeError(
        f"Expected 6940 output rows; wrote {rows_written}"
    )

if failures:
    raise RuntimeError(
        f"Extraction produced {len(failures)} hard failures"
    )

if audit_max_mean_difference > 1e-10:
    raise RuntimeError(
        "Extracted means disagree with prior audit: "
        f"{audit_max_mean_difference}"
    )

if audit_max_median_difference > 1e-10:
    raise RuntimeError(
        "Extracted medians disagree with prior audit: "
        f"{audit_max_median_difference}"
    )

if identity_max_mean_error > 1e-10:
    raise RuntimeError(
        f"Tract-mean tensor identity error: {identity_max_mean_error}"
    )

if identity_max_weighted_error > 1e-10:
    raise RuntimeError(
        "Weighted tract-mean tensor identity error: "
        f"{identity_max_weighted_error}"
    )

if primary_failures != 0:
    raise RuntimeError(
        f"Unexpected primary eligibility failures: {primary_failures}"
    )

if strict_coverage_failures != 1:
    raise RuntimeError(
        "Expected one strict coverage failure; found "
        f"{strict_coverage_failures}"
    )

if negative_5_failures != 12:
    raise RuntimeError(
        "Expected 12 cells at or above 5%; found "
        f"{negative_5_failures}"
    )

if negative_1_failures != 112:
    raise RuntimeError(
        "Expected 112 cells at or above 1%; found "
        f"{negative_1_failures}"
    )

if coarse_failures != 40:
    raise RuntimeError(
        "Expected 40 subject-tract cells from the two coarse "
        f"acquisitions; found {coarse_failures}"
    )


summary = f"""JHU TRACT-LEVEL FA/MD/AD/RD EXTRACTION

Subjects: {expected_subjects}
JHU probabilistic tracts: {expected_tracts}
Subject-tract rows: {rows_written}
Atlas probability threshold: {atlas_threshold:.0f}%
Skeleton threshold: 0.20

Statistics extracted for FA, MD, AD, and RD:
- Minimum and maximum
- Arithmetic mean
- Median
- Standard deviation
- First and third quartiles
- Interquartile range
- Atlas-probability-weighted mean
- Negative-voxel count and percentage

Validation:
- Hard extraction failures: {len(failures)}
- Primary coverage failures: {primary_failures}
- Strict 99% coverage failures: {strict_coverage_failures}
- Cells at or above 5% negative values: {negative_5_failures}
- Cells at or above 1% negative values: {negative_1_failures}
- Coarse-resolution cells flagged: {coarse_failures}
- Maximum prior-audit mean difference: {audit_max_mean_difference:.12g}
- Maximum prior-audit median difference: {audit_max_median_difference:.12g}
- Maximum tract-mean tensor identity error: {identity_max_mean_error:.12g}
- Maximum weighted-mean tensor identity error: {identity_max_weighted_error:.12g}

Interpretation:
- All finite values, including negative diffusivity values, were preserved.
- No voxel was clamped, replaced, deleted, or imputed.
- Eligibility flags were recorded without changing the measurements.
- ADNI and NIFD must be analyzed separately.
- ADNI site must be modeled where estimable.
"""

summary_file.write_text(summary)

status_file.write_text(
    "JHU TRACT EXTRACTION COMPLETE\n\n"
    f"Subjects: {expected_subjects}\n"
    f"Tracts: {expected_tracts}\n"
    f"Subject-tract rows: {rows_written}\n"
    "Metrics: FA, MD, AD, RD\n"
    "Primary eligible cells: 6940/6940\n"
    "Strict coverage eligible cells: 6939/6940\n"
    "Negative ≥5% sensitivity exclusions: 12\n"
    "Negative ≥1% sensitivity exclusions: 112\n"
    "Coarse-resolution sensitivity exclusions: 40 cells\n\n"
    "READY FOR TRACT TABLE VALIDATION AND METADATA AUDIT\n"
)

print(summary)
print(f"Saved: {output_file}")
print(f"Saved: {tract_file}")
print(f"Saved: {summary_file}")
print(f"Saved: {status_file}")
print(f"Saved: {failure_file}")
