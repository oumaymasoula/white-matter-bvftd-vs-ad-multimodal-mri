import csv
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST"])
out = Path(os.environ["OUT"])
tables = out / "tables"

with manifest_path.open(
    newline="",
    encoding="utf-8-sig",
) as handle:
    manifest = list(
        csv.DictReader(handle, delimiter="\t")
    )

eligible = [
    row
    for row in manifest
    if row["final_resolution_status"].startswith("PASS_")
]

failed = [
    row
    for row in manifest
    if not row["final_resolution_status"].startswith("PASS_")
]

if len(eligible) != 343:
    raise RuntimeError(
        f"Expected 343 extractable subjects; "
        f"found {len(eligible)}"
    )


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_aseg(path):
    global_measures = {}
    structures = {}

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for original in handle:
            line = original.strip()

            if line.startswith("# Measure "):
                content = line[len("# Measure "):]
                parts = [
                    item.strip()
                    for item in content.split(",")
                ]

                if len(parts) >= 4:
                    measure_id = parts[0]
                    measure_name = parts[1]
                    value = as_float(parts[-2])

                    global_measures[measure_id] = value
                    global_measures[measure_name] = value

                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # aseg.stats:
            # Index SegId NVoxels Volume_mm3 StructName ...
            if len(parts) >= 5:
                volume = as_float(parts[3])
                structure = parts[4]

                if math.isfinite(volume):
                    structures[structure] = volume

    return global_measures, structures


def parse_aparc(path, hemisphere):
    measures = {}
    regions = {}

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for original in handle:
            line = original.strip()

            if line.startswith("# Measure "):
                content = line[len("# Measure "):]
                parts = [
                    item.strip()
                    for item in content.split(",")
                ]

                if len(parts) >= 4:
                    measure_id = parts[0]
                    measure_name = parts[1]
                    value = as_float(parts[-2])

                    measures[measure_id] = value
                    measures[measure_name] = value

                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # aparc.stats:
            # StructName NumVert SurfArea GrayVol ThickAvg ...
            if len(parts) >= 10:
                region = parts[0]

                regions[region] = {
                    "hemisphere": hemisphere,
                    "num_vertices": as_float(parts[1]),
                    "surface_area_mm2": as_float(parts[2]),
                    "gray_volume_mm3": as_float(parts[3]),
                    "thickness_mean_mm": as_float(parts[4]),
                    "thickness_sd_mm": as_float(parts[5]),
                    "mean_curvature": as_float(parts[6]),
                    "gaussian_curvature": as_float(parts[7]),
                    "folding_index": as_float(parts[8]),
                    "curvature_index": as_float(parts[9]),
                }

    return measures, regions


def first_value(mapping, candidates):
    for candidate in candidates:
        value = mapping.get(candidate)

        if value is not None and math.isfinite(value):
            return value

    return float("nan")


def bilateral_sum(structures, left, right):
    left_value = structures.get(left, float("nan"))
    right_value = structures.get(right, float("nan"))

    if not (
        math.isfinite(left_value)
        and math.isfinite(right_value)
    ):
        return float("nan")

    return left_value + right_value


def bilateral_cortical(regions_lh, regions_rh, region):
    left = regions_lh.get(region)
    right = regions_rh.get(region)

    if left is None or right is None:
        return {
            "volume_mm3": float("nan"),
            "thickness_mm": float("nan"),
            "surface_area_mm2": float("nan"),
        }

    volume = (
        left["gray_volume_mm3"]
        + right["gray_volume_mm3"]
    )

    surface_area = (
        left["surface_area_mm2"]
        + right["surface_area_mm2"]
    )

    if surface_area > 0:
        thickness = (
            left["thickness_mean_mm"]
            * left["surface_area_mm2"]
            + right["thickness_mean_mm"]
            * right["surface_area_mm2"]
        ) / surface_area
    else:
        thickness = float("nan")

    return {
        "volume_mm3": volume,
        "thickness_mm": thickness,
        "surface_area_mm2": surface_area,
    }


def extract_fs_version(log_path):
    if not log_path.is_file():
        return ""

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    patterns = [
        r"freesurfer-linux-[^\s]+",
        r"ProgramVersion:\s*([^\s]+)",
        r"freesurfer\s+version\s+([^\s]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                match.group(1)
                if match.lastindex
                else match.group(0)
            )

    return "UNKNOWN"


target_cortical_regions = [
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "inferiortemporal",
    "insula",
    "lateralorbitofrontal",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiortemporal",
    "temporalpole",
]

subcortical_pairs = {
    "hippocampus": (
        "Left-Hippocampus",
        "Right-Hippocampus",
    ),
    "amygdala": (
        "Left-Amygdala",
        "Right-Amygdala",
    ),
    "thalamus": (
        "Left-Thalamus-Proper",
        "Right-Thalamus-Proper",
    ),
    "caudate": (
        "Left-Caudate",
        "Right-Caudate",
    ),
    "putamen": (
        "Left-Putamen",
        "Right-Putamen",
    ),
    "pallidum": (
        "Left-Pallidum",
        "Right-Pallidum",
    ),
    "accumbens": (
        "Left-Accumbens-area",
        "Right-Accumbens-area",
    ),
    "lateral_ventricle": (
        "Left-Lateral-Ventricle",
        "Right-Lateral-Ventricle",
    ),
    "inferior_lateral_ventricle": (
        "Left-Inf-Lat-Vent",
        "Right-Inf-Lat-Vent",
    ),
    "choroid_plexus": (
        "Left-choroid-plexus",
        "Right-choroid-plexus",
    ),
}

global_long_rows = []
aseg_long_rows = []
aparc_long_rows = []
wide_rows = []
audit_rows = []

for position, row in enumerate(eligible, start=1):
    directory = Path(
        row["canonical_subject_directory"]
    )

    aseg_path = directory / "stats/aseg.stats"
    lh_path = directory / "stats/lh.aparc.stats"
    rh_path = directory / "stats/rh.aparc.stats"
    log_path = directory / "scripts/recon-all.log"

    for required in [aseg_path, lh_path, rh_path]:
        if not required.is_file():
            raise RuntimeError(
                f"Missing statistics file: {required}"
            )

    globals_aseg, structures = parse_aseg(
        aseg_path
    )

    globals_lh, regions_lh = parse_aparc(
        lh_path,
        "lh",
    )

    globals_rh, regions_rh = parse_aparc(
        rh_path,
        "rh",
    )

    subject = row["subject"]
    dataset = row["dataset"]
    group = row["group"]

    for measure, value in sorted(
        globals_aseg.items()
    ):
        global_long_rows.append(
            {
                "dataset": dataset,
                "group": group,
                "subject": subject,
                "source": "aseg.stats",
                "measure": measure,
                "value": value,
            }
        )

    for structure, volume in sorted(
        structures.items()
    ):
        aseg_long_rows.append(
            {
                "dataset": dataset,
                "group": group,
                "subject": subject,
                "structure": structure,
                "volume_mm3": volume,
            }
        )

    for hemisphere, regions in [
        ("lh", regions_lh),
        ("rh", regions_rh),
    ]:
        for region, values in sorted(
            regions.items()
        ):
            aparc_long_rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "subject": subject,
                    "hemisphere": hemisphere,
                    "region": region,
                    **values,
                }
            )

    etiv = first_value(
        globals_aseg,
        [
            "EstimatedTotalIntraCranialVol",
            "eTIV",
            "IntraCranialVol",
        ],
    )

    feature = {
        "dataset": dataset,
        "group": group,
        "subject": subject,
        "freesurfer_subject_directory": str(
            directory
        ),
        "freesurfer_version": extract_fs_version(
            log_path
        ),
        "eTIV_mm3": etiv,
        "BrainSegVol_mm3": first_value(
            globals_aseg,
            ["BrainSegVol", "BrainSeg"],
        ),
        "BrainSegVolNotVent_mm3": first_value(
            globals_aseg,
            [
                "BrainSegVolNotVent",
                "BrainSegNotVent",
            ],
        ),
        "TotalGrayVol_mm3": first_value(
            globals_aseg,
            ["TotalGrayVol", "TotalGray"],
        ),
        "CortexVol_mm3": first_value(
            globals_aseg,
            ["CortexVol", "Cortex"],
        ),
        "CorticalWhiteMatterVol_mm3": first_value(
            globals_aseg,
            [
                "CorticalWhiteMatterVol",
                "CorticalWhiteMatter",
            ],
        ),
        "SubCortGrayVol_mm3": first_value(
            globals_aseg,
            ["SubCortGrayVol", "SubCortGray"],
        ),
        "SupraTentorialVol_mm3": first_value(
            globals_aseg,
            [
                "SupraTentorialVol",
                "SupraTentorial",
            ],
        ),
        "MaskVol_mm3": first_value(
            globals_aseg,
            ["MaskVol", "Mask"],
        ),
        "lh_MeanThickness_mm": first_value(
            globals_lh,
            ["MeanThickness", "ThickAvg"],
        ),
        "rh_MeanThickness_mm": first_value(
            globals_rh,
            ["MeanThickness", "ThickAvg"],
        ),
    }

    lh_mean = feature["lh_MeanThickness_mm"]
    rh_mean = feature["rh_MeanThickness_mm"]

    if math.isfinite(lh_mean) and math.isfinite(
        rh_mean
    ):
        feature["bilateral_MeanThickness_mm"] = (
            lh_mean + rh_mean
        ) / 2.0
    else:
        feature["bilateral_MeanThickness_mm"] = (
            float("nan")
        )

    for name, pair in subcortical_pairs.items():
        left_name, right_name = pair

        left_value = structures.get(
            left_name,
            float("nan"),
        )

        right_value = structures.get(
            right_name,
            float("nan"),
        )

        feature[f"lh_{name}_volume_mm3"] = left_value
        feature[f"rh_{name}_volume_mm3"] = right_value
        feature[f"bilateral_{name}_volume_mm3"] = (
            bilateral_sum(
                structures,
                left_name,
                right_name,
            )
        )

    wm_hypointensity_candidates = [
        "WM-hypointensities",
        "Left-WM-hypointensities",
        "Right-WM-hypointensities",
    ]

    feature["WM_hypointensities_volume_mm3"] = sum(
        structures.get(candidate, 0.0)
        for candidate in wm_hypointensity_candidates
    )

    corpus_callosum_names = [
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
    ]

    feature["corpus_callosum_volume_mm3"] = sum(
        structures.get(name, 0.0)
        for name in corpus_callosum_names
    )

    for region in target_cortical_regions:
        left = regions_lh.get(region)
        right = regions_rh.get(region)
        bilateral = bilateral_cortical(
            regions_lh,
            regions_rh,
            region,
        )

        for hemisphere, values in [
            ("lh", left),
            ("rh", right),
        ]:
            feature[
                f"{hemisphere}_{region}_volume_mm3"
            ] = (
                values["gray_volume_mm3"]
                if values
                else float("nan")
            )

            feature[
                f"{hemisphere}_{region}_thickness_mm"
            ] = (
                values["thickness_mean_mm"]
                if values
                else float("nan")
            )

            feature[
                f"{hemisphere}_{region}_area_mm2"
            ] = (
                values["surface_area_mm2"]
                if values
                else float("nan")
            )

        feature[
            f"bilateral_{region}_volume_mm3"
        ] = bilateral["volume_mm3"]

        feature[
            f"bilateral_{region}_thickness_mm"
        ] = bilateral["thickness_mm"]

        feature[
            f"bilateral_{region}_area_mm2"
        ] = bilateral["surface_area_mm2"]

    wide_rows.append(feature)

    essential_values = {
        "eTIV_mm3": feature["eTIV_mm3"],
        "CortexVol_mm3": feature["CortexVol_mm3"],
        "hippocampus": feature[
            "bilateral_hippocampus_volume_mm3"
        ],
        "amygdala": feature[
            "bilateral_amygdala_volume_mm3"
        ],
        "mean_thickness": feature[
            "bilateral_MeanThickness_mm"
        ],
    }

    missing_essential = [
        name
        for name, value in essential_values.items()
        if not math.isfinite(value)
    ]

    missing_target_regions = [
        region
        for region in target_cortical_regions
        if (
            region not in regions_lh
            or region not in regions_rh
        )
    ]

    audit_rows.append(
        {
            "dataset": dataset,
            "group": group,
            "subject": subject,
            "canonical_subject_directory": str(
                directory
            ),
            "freesurfer_version": feature[
                "freesurfer_version"
            ],
            "aseg_structure_count": len(structures),
            "lh_aparc_region_count": len(regions_lh),
            "rh_aparc_region_count": len(regions_rh),
            "missing_essential_count": len(
                missing_essential
            ),
            "missing_essential": ";".join(
                missing_essential
            ),
            "missing_target_region_count": len(
                missing_target_regions
            ),
            "missing_target_regions": ";".join(
                missing_target_regions
            ),
            "extraction_status": (
                "PASS"
                if (
                    not missing_essential
                    and not missing_target_regions
                )
                else "FAIL_MISSING_FEATURES"
            ),
        }
    )

    if position % 50 == 0 or position == len(eligible):
        print(
            f"Extracted {position}/{len(eligible)}"
        )


def write_table(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows generated for {path.name}"
        )

    fieldnames = list(rows[0])

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


global_path = (
    tables / "FreeSurfer_global_measures_long.tsv"
)

aseg_path_out = (
    tables / "FreeSurfer_aseg_structures_long.tsv"
)

aparc_path_out = (
    tables / "FreeSurfer_aparc_regions_long.tsv"
)

wide_path = (
    tables / "FreeSurfer_targeted_features_wide.tsv"
)

audit_path = (
    tables / "FreeSurfer_extraction_audit.tsv"
)

failed_path = (
    tables / "FreeSurfer_failed_subjects.tsv"
)

write_table(global_path, global_long_rows)
write_table(aseg_path_out, aseg_long_rows)
write_table(aparc_path_out, aparc_long_rows)
write_table(wide_path, wide_rows)
write_table(audit_path, audit_rows)
write_table(failed_path, failed)

audit_counts = Counter(
    row["extraction_status"]
    for row in audit_rows
)

version_counts = Counter(
    row["freesurfer_version"]
    for row in audit_rows
)

dataset_group_counts = Counter(
    (row["dataset"], row["group"])
    for row in audit_rows
    if row["extraction_status"] == "PASS"
)

summary_lines = [
    "FREESURFER FEATURE EXTRACTION",
    "",
    f"Eligible subjects: {len(eligible)}",
    f"Wide feature rows: {len(wide_rows)}",
    f"Failed computational subjects: {len(failed)}",
    "",
    "EXTRACTION STATUS",
]

for status, count in sorted(audit_counts.items()):
    summary_lines.append(
        f"{status}\t{count}"
    )

summary_lines.extend(
    [
        "",
        "PASS COUNTS BY DATASET AND GROUP",
    ]
)

for key, count in sorted(
    dataset_group_counts.items()
):
    dataset, group = key
    summary_lines.append(
        f"{dataset}\t{group}\t{count}"
    )

summary_lines.extend(
    [
        "",
        "FREESURFER VERSIONS",
    ]
)

for version, count in sorted(
    version_counts.items()
):
    summary_lines.append(
        f"{version}\t{count}"
    )

summary_lines.extend(
    [
        "",
        "OUTPUTS",
        (
            "Global measures, all aseg structures, all aparc "
            "regions, targeted wide features and extraction audit."
        ),
        "",
        "IMPORTANT",
        (
            "Extraction completion does not replace anatomical "
            "visual QC."
        ),
        (
            "Regional volumes must be modelled with eTIV; "
            "cortical thickness must not be divided by eTIV."
        ),
    ]
)

summary_path = (
    out / "FreeSurfer_feature_extraction_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

checksum_path = out / "SHA256SUMS.txt"

files = [
    global_path,
    aseg_path_out,
    aparc_path_out,
    wide_path,
    audit_path,
    failed_path,
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

print("\n".join(summary_lines))

if audit_counts.get("FAIL_MISSING_FEATURES", 0):
    raise RuntimeError(
        "Some subjects are missing essential FreeSurfer features"
    )
