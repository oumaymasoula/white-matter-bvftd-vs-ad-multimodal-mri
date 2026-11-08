import csv
import math
import sys
from collections import Counter
from pathlib import Path

demographic_path = Path(sys.argv[1])
tract_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])


def read_tsv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames

    if not fields:
        raise RuntimeError(f"Missing header: {path}")

    return fields, rows


def clean(value):
    return "" if value is None else str(value).strip()


demo_fields, demo_rows = read_tsv(demographic_path)
tract_fields, tract_rows = read_tsv(tract_path)

if len(demo_rows) != 347:
    raise RuntimeError(
        f"Demographic rows={len(demo_rows)}, expected 347"
    )

if len(tract_rows) != 6940:
    raise RuntimeError(
        f"Tract rows={len(tract_rows)}, expected 6940"
    )


# ------------------------------------------------------------
# Index one demographic record per task
# ------------------------------------------------------------

demo_by_task = {}

for row in demo_rows:
    task = int(clean(row.get("task")))

    if task in demo_by_task:
        raise RuntimeError(
            f"Duplicate demographic task: {task}"
        )

    demo_by_task[task] = row

if set(demo_by_task) != set(range(1, 348)):
    raise RuntimeError(
        "Demographic task numbers are not exactly 1–347"
    )


# ------------------------------------------------------------
# Metadata appended to each tract record
# ------------------------------------------------------------

appended_fields = [
    "sex",
    "age_at_DWI_years",
    "selected_DWI_date",
    "selected_DTI_image_id",
    "demographic_visit_date",
    "demographic_days_from_DWI",
    "demographic_visit",
    "acquisition_site",
    "manufacturer",
    "scanner_model",
    "field_strength_T",
    "series_description",
    "age_source",
    "age_method",
    "age_precision",
    "sex_source",
    "demographic_match_status",
]

collisions = [
    field for field in appended_fields
    if field in tract_fields
]

if collisions:
    raise RuntimeError(
        f"Column collision with tract table: {collisions}"
    )

output_fields = tract_fields + appended_fields
output_rows = []
tract_count = Counter()

for tract in tract_rows:
    task = int(clean(tract.get("task")))
    demo = demo_by_task.get(task)

    if demo is None:
        raise RuntimeError(
            f"Missing demographics for task {task}"
        )

    comparisons = [
        (
            "dataset",
            clean(tract.get("dataset")),
            clean(demo.get("dataset")),
        ),
        (
            "group",
            clean(tract.get("group")),
            clean(demo.get("analysis_group")),
        ),
        (
            "subject",
            clean(tract.get("subject")),
            clean(demo.get("subject_bids")),
        ),
        (
            "volume index",
            clean(tract.get("volume_index_zero_based")),
            clean(demo.get("volume_index_zero_based")),
        ),
    ]

    for label, tract_value, demo_value in comparisons:
        if tract_value != demo_value:
            raise RuntimeError(
                f"Identity mismatch task {task}, {label}: "
                f"{tract_value!r} != {demo_value!r}"
            )

    sex = clean(demo.get("sex"))
    age_text = clean(demo.get("age_at_DWI_years"))

    if sex not in {"M", "F"}:
        raise RuntimeError(
            f"Invalid sex task {task}: {sex!r}"
        )

    try:
        age = float(age_text)
    except ValueError:
        raise RuntimeError(
            f"Invalid age task {task}: {age_text!r}"
        )

    if not math.isfinite(age) or not 35 <= age <= 100:
        raise RuntimeError(
            f"Implausible age task {task}: {age}"
        )

    joined = dict(tract)

    joined.update(
        {
            "sex": sex,
            "age_at_DWI_years": age_text,
            "selected_DWI_date": clean(
                demo.get("selected_DWI_date")
            ),
            "selected_DTI_image_id": clean(
                demo.get("selected_DTI_image_id")
            ),
            "demographic_visit_date": clean(
                demo.get("demographic_visit_date")
            ),
            "demographic_days_from_DWI": clean(
                demo.get("demographic_days_from_DWI")
            ),
            "demographic_visit": clean(
                demo.get("demographic_visit")
            ),
            "acquisition_site": clean(
                demo.get("acquisition_site")
            ),
            "manufacturer": clean(
                demo.get("manufacturer")
            ),
            "scanner_model": clean(
                demo.get("scanner_model")
            ),
            "field_strength_T": clean(
                demo.get("field_strength_T")
            ),
            "series_description": clean(
                demo.get("series_description")
            ),
            "age_source": clean(
                demo.get("age_source")
            ),
            "age_method": clean(
                demo.get("age_method")
            ),
            "age_precision": clean(
                demo.get("age_precision")
            ),
            "sex_source": clean(
                demo.get("sex_source")
            ),
            "demographic_match_status": clean(
                demo.get("source_match_status")
            ),
        }
    )

    # Confirm every original tract value was preserved.
    for field in tract_fields:
        if joined[field] != tract[field]:
            raise RuntimeError(
                f"Original value changed: task={task}, "
                f"field={field}"
            )

    output_rows.append(joined)
    tract_count[task] += 1


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

if len(output_rows) != 6940:
    raise RuntimeError(
        f"Joined rows={len(output_rows)}, expected 6940"
    )

bad_counts = {
    task: count
    for task, count in tract_count.items()
    if count != 20
}

if bad_counts:
    raise RuntimeError(
        f"Subjects without exactly 20 tracts: {bad_counts}"
    )

cell_keys = [
    (
        row["task"],
        row["tract_index_zero_based"],
    )
    for row in output_rows
]

if len(set(cell_keys)) != 6940:
    raise RuntimeError(
        "Duplicate task-by-tract cells detected"
    )

for field in (
    "sex",
    "age_at_DWI_years",
    "selected_DWI_date",
    "selected_DTI_image_id",
    "acquisition_site",
):
    missing = sum(
        not clean(row.get(field))
        for row in output_rows
    )

    if missing:
        raise RuntimeError(
            f"Missing {field}: {missing} rows"
        )


# ------------------------------------------------------------
# Save joined table
# ------------------------------------------------------------

with output_path.open(
    "w", encoding="utf-8", newline=""
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=output_fields,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output_rows)


group_counts = Counter(
    (
        row["dataset"],
        row["analysis_group"],
    )
    for row in demo_rows
)

sex_counts = Counter(
    (
        row["dataset"],
        row["analysis_group"],
        row["sex"],
    )
    for row in demo_rows
)

ages = [
    float(row["age_at_DWI_years"])
    for row in demo_rows
]

summary = [
    "JHU TRACT AND DEMOGRAPHIC JOIN",
    "",
    "Subjects: 347",
    "Tracts per subject: 20",
    "Subject-tract rows: 6940",
    "Demographic matches: 347/347",
    "Sex complete: 347/347",
    "Age complete: 347/347",
    "Selected DWI dates complete: 347/347",
    "Selected DTI image IDs complete: 347/347",
    "Acquisition sites complete: 347/347",
    "Original tract measurements preserved: YES",
    f"Age range: {min(ages):.4f} to {max(ages):.4f}",
    "",
    "Dataset/group counts:",
]

for key in sorted(group_counts):
    summary.append(
        f"- {key[0]}/{key[1]}: {group_counts[key]}"
    )

summary.extend(["", "Dataset/group/sex counts:"])

for key in sorted(sex_counts):
    summary.append(
        f"- {key[0]}/{key[1]}/{key[2]}: "
        f"{sex_counts[key]}"
    )

summary.extend(
    [
        "",
        "Analysis policy:",
        "- Analyze NIFD and ADNI separately.",
        "- NIFD comparison: bvFTD versus CN.",
        "- ADNI primary omnibus comparison: CN, MCI and AD.",
        "- ADNI planned contrasts: MCI-CN, AD-CN and AD-MCI.",
        "- Adjust for age and sex.",
        "- Model acquisition site only after estimability audit.",
        "- Do not pool ADNI and NIFD controls.",
        "",
        "No DOB or birth-month value was included.",
        "No tract measurement was modified.",
        "",
        "READY FOR DEMOGRAPHIC AND SITE-CONFOUNDING AUDIT",
    ]
)

summary_path.write_text(
    "\n".join(summary) + "\n",
    encoding="utf-8",
)

print("\n".join(summary))
print(f"\nSaved: {output_path}")
print(f"Saved: {summary_path}")
