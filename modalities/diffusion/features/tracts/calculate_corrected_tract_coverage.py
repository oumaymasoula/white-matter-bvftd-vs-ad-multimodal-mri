import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

project = Path(os.environ["PROJECT"])
tbssroot = Path(os.environ["TBSSROOT"])
projqc = Path(os.environ["PROJQC"])
support_root = Path(os.environ["SUPPORT"])
manifest = Path(os.environ["MANIFEST"])
atlas_path = Path(os.environ["ATLAS"])
xml_path = Path(os.environ["XML"])
skeleton_path = Path(os.environ["SKELETON"])

subjects = []

for task, line in enumerate(manifest.read_text().splitlines(), start=1):
    fields = line.split("\t")

    if len(fields) < 3:
        raise RuntimeError(f"Malformed manifest line {task}: {line!r}")

    dataset, group, subject = fields[:3]
    name = f"t{task:04d}__{dataset}__{group}__{subject}_FA_FA"

    subjects.append({
        "task": task,
        "dataset": dataset,
        "group": group,
        "subject": subject,
        "name": name,
    })

if len(subjects) != 347:
    raise RuntimeError(f"Expected 347 subjects, found {len(subjects)}")

skeleton_img = nib.load(skeleton_path)
atlas_img = nib.load(atlas_path)

skeleton = np.asarray(skeleton_img.dataobj) > 0
atlas = np.asarray(atlas_img.dataobj, dtype=np.float32)

if skeleton.shape != (182, 218, 182):
    raise RuntimeError(f"Unexpected skeleton shape: {skeleton.shape}")

if atlas.shape[:3] != skeleton.shape:
    raise RuntimeError(
        f"Atlas spatial geometry mismatch: {atlas.shape[:3]} "
        f"vs {skeleton.shape}"
    )

if atlas.ndim != 4:
    raise RuntimeError(
        f"Expected a 4D probabilistic tract atlas; found {atlas.shape}"
    )

if not np.allclose(
    skeleton_img.affine, atlas_img.affine, atol=1e-4
):
    raise RuntimeError("Atlas and skeleton affines do not match")

tree = ET.parse(xml_path)
xml_labels = {}

for element in tree.findall(".//label"):
    index_text = element.attrib.get("index")

    if index_text is None:
        continue

    index = int(index_text)
    label = " ".join((element.text or "").split())
    xml_labels[index] = label

number_of_tracts = atlas.shape[3]

if number_of_tracts != 20:
    raise RuntimeError(
        f"Expected 20 JHU probabilistic tracts; found {number_of_tracts}"
    )

skeleton_voxels = int(skeleton.sum())

if skeleton_voxels != 114436:
    raise RuntimeError(
        f"Unexpected skeleton voxel count: {skeleton_voxels}"
    )

tracts = []
roi_inventory = []

for volume_index in range(number_of_tracts):
    probability = atlas[..., volume_index]
    roi = skeleton & (probability >= 25.0)
    voxel_count = int(roi.sum())
    probability_sum = float(probability[roi].sum())

    label = xml_labels.get(
        volume_index,
        f"UNLABELLED_TRACT_VOLUME_{volume_index}",
    )

    tracts.append({
        "volume_index": volume_index,
        "tract": label,
        "probability": probability,
        "roi": roi,
        "skeleton_voxels": voxel_count,
        "probability_sum": probability_sum,
    })

    roi_inventory.append({
        "volume_index": volume_index,
        "tract": label,
        "probability_threshold_percent": 25,
        "skeleton_voxels": voxel_count,
        "probability_sum": probability_sum,
        "roi_size_review": (
            "INSUFFICIENT_LT_30_VOXELS"
            if voxel_count < 30
            else "SMALL_30_TO_99_VOXELS"
            if voxel_count < 100
            else "ADEQUATE_AT_LEAST_100_VOXELS"
        ),
    })

inventory_fields = [
    "volume_index",
    "tract",
    "probability_threshold_percent",
    "skeleton_voxels",
    "probability_sum",
    "roi_size_review",
]

with (
    projqc / "jhu_tract_skeleton_roi_inventory.tsv"
).open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        delimiter="\t",
        fieldnames=inventory_fields,
    )
    writer.writeheader()
    writer.writerows(roi_inventory)

global_rows = []
tract_rows = []

for subject in subjects:
    mask_path = (
        support_root / f"{subject['name']}_support.nii.gz"
    )

    if not mask_path.is_file() or mask_path.stat().st_size == 0:
        raise RuntimeError(f"Missing support mask: {mask_path}")

    mask_img = nib.load(mask_path)

    if mask_img.shape != skeleton.shape:
        raise RuntimeError(
            f"Geometry mismatch for {subject['name']}: "
            f"{mask_img.shape}"
        )

    if not np.allclose(
        mask_img.affine, skeleton_img.affine, atol=1e-4
    ):
        raise RuntimeError(
            f"Affine mismatch for {subject['name']}"
        )

    support = np.asarray(mask_img.dataobj) > 0.5

    supported_global = int(np.count_nonzero(skeleton & support))
    global_coverage = (
        100.0 * supported_global / skeleton_voxels
    )

    global_rows.append({
        **subject,
        "skeleton_voxels": skeleton_voxels,
        "supported_skeleton_voxels": supported_global,
        "unsupported_skeleton_voxels": (
            skeleton_voxels - supported_global
        ),
        "global_support_percent": global_coverage,
    })

    for tract in tracts:
        roi = tract["roi"]
        total = tract["skeleton_voxels"]

        if total == 0:
            covered = 0
            unweighted_coverage = np.nan
            weighted_coverage = np.nan
        else:
            covered_mask = roi & support
            covered = int(covered_mask.sum())
            unweighted_coverage = 100.0 * covered / total

            denominator = tract["probability_sum"]

            if denominator > 0:
                numerator = float(
                    tract["probability"][covered_mask].sum()
                )
                weighted_coverage = 100.0 * numerator / denominator
            else:
                weighted_coverage = np.nan

        tract_rows.append({
            **subject,
            "tract_volume_index": tract["volume_index"],
            "tract": tract["tract"],
            "tract_skeleton_voxels": total,
            "supported_tract_voxels": covered,
            "unsupported_tract_voxels": total - covered,
            "unweighted_support_percent": unweighted_coverage,
            "probability_weighted_support_percent": weighted_coverage,
        })

global_fields = [
    "task", "dataset", "group", "subject", "name",
    "skeleton_voxels",
    "supported_skeleton_voxels",
    "unsupported_skeleton_voxels",
    "global_support_percent",
]

with (
    projqc / "corrected_global_support_coverage.tsv"
).open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        delimiter="\t",
        fieldnames=global_fields,
    )
    writer.writeheader()
    writer.writerows(global_rows)

tract_fields = [
    "task", "dataset", "group", "subject", "name",
    "tract_volume_index", "tract",
    "tract_skeleton_voxels",
    "supported_tract_voxels",
    "unsupported_tract_voxels",
    "unweighted_support_percent",
    "probability_weighted_support_percent",
]

with (
    projqc / "corrected_subject_tract_coverage.tsv"
).open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        delimiter="\t",
        fieldnames=tract_fields,
    )
    writer.writeheader()
    writer.writerows(tract_rows)

global_values = np.array(
    [row["global_support_percent"] for row in global_rows],
    dtype=float,
)

tract_values = np.array(
    [
        row["unweighted_support_percent"]
        for row in tract_rows
        if np.isfinite(row["unweighted_support_percent"])
    ],
    dtype=float,
)

weighted_values = np.array(
    [
        row["probability_weighted_support_percent"]
        for row in tract_rows
        if np.isfinite(
            row["probability_weighted_support_percent"]
        )
    ],
    dtype=float,
)

subject_minimum_rows = []

for subject in subjects:
    rows = [
        row for row in tract_rows
        if row["task"] == subject["task"]
        and np.isfinite(row["unweighted_support_percent"])
    ]

    minimum_row = min(
        rows,
        key=lambda row: row["unweighted_support_percent"],
    )

    subject_minimum_rows.append(minimum_row)

subject_minima = np.array(
    [
        row["unweighted_support_percent"]
        for row in subject_minimum_rows
    ],
    dtype=float,
)

lowest_global = sorted(
    global_rows,
    key=lambda row: row["global_support_percent"],
)

lowest_tract = sorted(
    tract_rows,
    key=lambda row: (
        np.inf
        if not np.isfinite(row["unweighted_support_percent"])
        else row["unweighted_support_percent"]
    ),
)

with (
    projqc / "corrected_lowest_global_coverage.tsv"
).open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        delimiter="\t",
        fieldnames=global_fields,
    )
    writer.writeheader()
    writer.writerows(lowest_global)

with (
    projqc / "corrected_lowest_subject_tract_coverage.tsv"
).open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        delimiter="\t",
        fieldnames=tract_fields,
    )
    writer.writeheader()
    writer.writerows(lowest_tract)

def distribution(values):
    return (
        f"min={np.min(values):.6f}, "
        f"p05={np.percentile(values,5):.6f}, "
        f"p25={np.percentile(values,25):.6f}, "
        f"median={np.median(values):.6f}, "
        f"p75={np.percentile(values,75):.6f}, "
        f"max={np.max(values):.6f}"
    )

small_lt30 = sum(
    row["skeleton_voxels"] < 30 for row in roi_inventory
)

small_lt100 = sum(
    row["skeleton_voxels"] < 100 for row in roi_inventory
)

summary = [
    "CORRECTED TBSS ACQUISITION-SUPPORT COVERAGE AUDIT",
    "",
    f"Subjects: {len(subjects)}",
    f"Skeleton voxels: {skeleton_voxels}",
    f"JHU probabilistic tracts: {number_of_tracts}",
    f"Subject-tract records: {len(tract_rows)}",
    "JHU probability threshold: 25%",
    "",
    "Global skeleton support:",
    distribution(global_values),
    "",
    "Unweighted subject-tract support:",
    distribution(tract_values),
    "",
    "Probability-weighted subject-tract support:",
    distribution(weighted_values),
    "",
    "Minimum tract support within each subject:",
    distribution(subject_minima),
    "",
    f"Tract ROIs with fewer than 30 skeleton voxels: {small_lt30}",
    f"Tract ROIs with fewer than 100 skeleton voxels: {small_lt100}",
    "",
    "No subject or tract was excluded.",
    "No coverage eligibility threshold was applied.",
    "ROI size and coverage rules require final definition.",
]

(projqc / "corrected_coverage_summary.txt").write_text(
    "\n".join(summary) + "\n"
)

print("\n".join(summary))

print("\nJHU tract ROI inventory:")
for row in roi_inventory:
    print(
        f"{row['volume_index']:2d}\t"
        f"{row['skeleton_voxels']:6d}\t"
        f"{row['roi_size_review']}\t"
        f"{row['tract']}"
    )

print("\nLowest 20 global-support subjects:")
for rank, row in enumerate(lowest_global[:20], start=1):
    print(
        f"{rank:2d}. "
        f"{row['dataset']}/{row['group']}/{row['subject']}: "
        f"{row['global_support_percent']:.6f}%"
    )

print("\nLowest 30 subject-tract support records:")
for rank, row in enumerate(lowest_tract[:30], start=1):
    print(
        f"{rank:2d}. "
        f"{row['dataset']}/{row['group']}/{row['subject']} | "
        f"{row['tract']} | "
        f"voxels={row['tract_skeleton_voxels']} | "
        f"coverage={row['unweighted_support_percent']:.6f}% | "
        f"weighted="
        f"{row['probability_weighted_support_percent']:.6f}%"
    )
