import csv
import hashlib
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

out = Path(os.environ["OUT"])
paths = {name: Path(os.environ[name]) for name in ("ALPS", "WMH", "DTI", "FS")}

def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not rows:
        raise RuntimeError(f"Empty table: {path}")
    return fields, rows

def write_table(name, rows, fields):
    with (out / "tables" / name).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def key(row):
    return (row["dataset"].strip(), row["subject"].strip())

def normalized_group(value):
    value = value.strip()
    return "AD" if value == "Probable_AD" else value

def true_flag(value):
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "pass"}

tables = {}
headers = {}
schema_rows = []

for name, path in paths.items():
    fields, rows = read_table(path)
    required = {"dataset", "group", "subject"}
    if not required.issubset(fields):
        raise RuntimeError(
            f"{name}: missing identity columns {sorted(required-set(fields))}"
        )

    headers[name] = fields
    tables[name] = rows
    for position, field in enumerate(fields, 1):
        schema_rows.append({
            "source": name,
            "position": position,
            "column": field,
        })

write_table("input_column_inventory.tsv", schema_rows,
            ["source", "position", "column"])

# Select explicit measurement columns; never silently substitute
# probability-weighted means for unweighted means.
alps_col = "ALPS_bilateral_mean_of_ratios"
wmh_col = "wmh_volume_ml"
fs_cols = [
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
dti_cols = [f"{metric}_mean" for metric in ("FA", "MD", "RD", "AD")]

for name, required in {
    "ALPS": [alps_col],
    "WMH": [wmh_col],
    "FS": fs_cols,
    "DTI": ["tract", "primary_analysis_eligible"] + dti_cols,
}.items():
    missing = set(required) - set(headers[name])
    if missing:
        raise RuntimeError(f"{name}: missing columns {sorted(missing)}")

# Audit unweighted means only. Final extraction definition must match
# the primary DTI source models before any statistical refitting.
maps = {}
for name in ("ALPS", "WMH", "FS"):
    mapping = {}
    for row in tables[name]:
        k = key(row)
        if k in mapping:
            raise RuntimeError(f"{name}: duplicate subject key {k}")
        mapping[k] = row
    maps[name] = mapping

dti_map = defaultdict(list)
seen = set()
for row in tables["DTI"]:
    k = key(row)
    tract = row["tract"].strip()
    tk = k + (tract,)
    if tk in seen:
        raise RuntimeError(f"Duplicate DTI subject/tract: {tk}")
    seen.add(tk)
    dti_map[k].append(row)

expected_tracts = sorted({
    row["tract"].strip() for row in tables["DTI"]
})
if len(expected_tracts) != 20:
    raise RuntimeError(
        f"Expected the 20-tract input; found {len(expected_tracts)} tracts. "
        "Check the source before proceeding."
    )

# WMH membership is the documented 347-participant source roster.
roster = maps["WMH"]
if len(roster) != 347:
    raise RuntimeError(f"Expected 347 source participants; found {len(roster)}")

flag_rows = []
audit_rows = []
candidate_rows = []
label_conflicts = []
extra_rows = []

for name, mapping in maps.items():
    for k in sorted(set(mapping) - set(roster)):
        extra_rows.append({"source": name, "dataset": k[0], "subject": k[1]})
for k in sorted(set(dti_map) - set(roster)):
    extra_rows.append({"source": "DTI", "dataset": k[0], "subject": k[1]})

# Preserve existing QC/support flags rather than guessing their meaning.
flag_columns = {
    name: [
        col for col in headers[name]
        if any(term in col.lower() for term in
               ("qc", "eligible", "included", "status", "exclusion"))
    ]
    for name in tables
}

for k in sorted(roster):
    base = roster[k]
    group = normalized_group(base["group"])
    reasons = []
    record = {"dataset": k[0], "group": group, "subject": k[1]}

    for name in ("ALPS", "WMH", "FS"):
        row = maps[name].get(k)
        record[f"has_{name}"] = int(row is not None)
        if row is None:
            reasons.append(f"missing_{name}_row")
            continue

        if normalized_group(row["group"]) != group:
            label_conflicts.append({
                "dataset": k[0], "subject": k[1],
                "source": name, "roster_group": group,
                "source_group": row["group"],
            })
            reasons.append(f"{name}_group_conflict")

        for col in flag_columns[name]:
            flag_rows.append({
                "dataset": k[0], "group": group, "subject": k[1],
                "source": name, "tract": "",
                "field": col, "value": row.get(col, ""),
            })

    alps = maps["ALPS"].get(k, {})
    fs = maps["FS"].get(k, {})
    wm = maps["WMH"][k]

    if alps and not finite(alps.get(alps_col)):
        reasons.append("nonfinite_bilateral_ALPS")

    if not finite(wm.get(wmh_col)):
        reasons.append("nonfinite_WMH_volume")
    elif float(wm[wmh_col]) < 0:
        reasons.append("negative_WMH_volume")
    elif float(wm[wmh_col]) == 0:
        # These project masks are unresolved, not automatically lesion-free.
        reasons.append("empty_WMH_mask_requires_review")

    missing_fs = [col for col in fs_cols if not finite(fs.get(col))]
    if fs and missing_fs:
        reasons.append("nonfinite_FS_features")
    if fs and finite(fs.get("eTIV_ml")) and float(fs["eTIV_ml"]) <= 0:
        reasons.append("invalid_eTIV")
    record["FS_missing_features"] = ";".join(missing_fs)

    drows = dti_map.get(k, [])
    actual_tracts = {row["tract"].strip() for row in drows}
    record["DTI_tract_count"] = len(actual_tracts)
    if actual_tracts != set(expected_tracts):
        reasons.append("incomplete_DTI_tract_set")

    bad_numeric = []
    ineligible = []
    for row in drows:
        tract = row["tract"].strip()
        if normalized_group(row["group"]) != group:
            label_conflicts.append({
                "dataset": k[0], "subject": k[1],
                "source": f"DTI:{tract}", "roster_group": group,
                "source_group": row["group"],
            })
            reasons.append("DTI_group_conflict")

        for col in dti_cols:
            if not finite(row.get(col)):
                bad_numeric.append(f"{tract}:{col}")

        if not true_flag(row.get("primary_analysis_eligible")):
            ineligible.append(tract)

        for col in flag_columns["DTI"]:
            flag_rows.append({
                "dataset": k[0], "group": group, "subject": k[1],
                "source": "DTI", "tract": tract,
                "field": col, "value": row.get(col, ""),
            })

    if bad_numeric:
        reasons.append("nonfinite_DTI_means")
    if ineligible:
        reasons.append("DTI_primary_ineligible_tracts")

    record["DTI_nonfinite_measurements"] = ";".join(bad_numeric)
    record["DTI_ineligible_tracts"] = ";".join(ineligible)
    record["candidate_common_measurement_sample"] = int(not reasons)
    record["not_candidate_reasons"] = ";".join(sorted(set(reasons)))

    # Copy support indicators without treating them as anatomical QC.
    for col in ("included_primary_full_cohort",
                "included_shared_site_sensitivity", "empty_mask_status"):
        record[f"WMH_{col}"] = wm.get(col, "COLUMN_NOT_PRESENT")

    audit_rows.append(record)

    if not reasons:
        candidate_rows.append({
            "dataset": k[0], "group": group, "subject": k[1],
            "ALPS_bilateral_mean_of_ratios": alps[alps_col],
            "wmh_volume_ml": wm[wmh_col],
            **{col: fs[col] for col in fs_cols},
            "status": "CANDIDATE_PENDING_QC_AND_SUPPORT_RECONCILIATION",
        })

write_table("all_subjects_common_sample_audit.tsv",
            audit_rows, list(audit_rows[0]))
write_table("candidate_common_measurement_sample.tsv",
            candidate_rows,
            ["dataset", "group", "subject", alps_col, wmh_col]
            + fs_cols + ["status"])
write_table("existing_QC_and_eligibility_flags.tsv", flag_rows,
            ["dataset", "group", "subject", "source", "tract", "field", "value"])
write_table("diagnostic_label_conflicts.tsv", label_conflicts,
            ["dataset", "subject", "source", "roster_group", "source_group"])
write_table("source_subjects_outside_roster.tsv", extra_rows,
            ["source", "dataset", "subject"])

# Site-by-group support among candidates; never infer it from a filename.
site_counts = defaultdict(Counter)
for row in candidate_rows:
    k = (row["dataset"], row["subject"])
    a = maps["ALPS"][k]
    w = maps["WMH"][k]
    site = str(a.get("acquisition_site") or a.get("site") or
               w.get("site") or "MISSING").strip()
    site_counts[(k[0], site)][row["group"]] += 1

site_rows = []
for (dataset, site), counts in sorted(site_counts.items()):
    groups = ["CN", "bvFTD"] if dataset == "NIFD" else ["CN", "MCI", "AD"]
    site_rows.append({
        "dataset": dataset, "site": site,
        "n_CN": counts["CN"], "n_bvFTD": counts["bvFTD"],
        "n_MCI": counts["MCI"], "n_AD": counts["AD"],
        "all_relevant_groups_present": int(
            site != "MISSING" and all(counts[g] > 0 for g in groups)
        ),
    })
write_table("candidate_site_group_support.tsv", site_rows,
            ["dataset", "site", "n_CN", "n_bvFTD", "n_MCI", "n_AD",
             "all_relevant_groups_present"])

summary = [
    "COMMON01: FULL MULTIMODAL COHORT AUDIT",
    "",
    f"Source participants: {len(roster)}",
    f"Candidate common measurement sample: {len(candidate_rows)}",
    f"Diagnostic-label conflicts: {len(label_conflicts)}",
    "",
    "CANDIDATES BY DATASET AND GROUP",
]
for (dataset, group), count in sorted(Counter(
    (r["dataset"], r["group"]) for r in candidate_rows
).items()):
    summary.append(f"{dataset}\t{group}\t{count}")

summary += ["", "REASONS FOR NOT ENTERING CANDIDATE SAMPLE"]
reason_counts = Counter(
    reason
    for row in audit_rows
    for reason in row["not_candidate_reasons"].split(";")
    if reason
)
for reason, count in reason_counts.most_common():
    summary.append(f"{reason}\t{count}")

summary += [
    "",
    "INTERPRETATION",
    "This is a candidate measurement-complete sample, not a final QC-approved cohort.",
    "All 20 DTI tracts and all four unweighted metric means were checked.",
    "The final DTI extraction statistic must match the source primary models.",
    "All 15 selected FreeSurfer fields, bilateral ALPS and WMH volume were checked.",
    "Empty WMH masks remain unresolved; this audit does not assign anatomical failure.",
    "Site/protocol eligibility and completed anatomical QC must be reconciled next.",
    "No regression was fitted and no existing measurement was changed.",
]
text = "\n".join(summary) + "\n"
(out / "COMMON01_summary.txt").write_text(text, encoding="utf-8")
print(text)

with (out / "input_SHA256SUMS.txt").open("w") as handle:
    for name, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path}\n")

if label_conflicts:
    raise RuntimeError(
        "Diagnostic-label conflicts were saved. Resolve before any refitting."
    )
