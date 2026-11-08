from pathlib import Path
from datetime import datetime, date
import re
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel

ROOT = Path.home() / "scratch/WM_Project/clinical_analysis"
OUT = ROOT / "outputs/tables"
AUDIT = ROOT / "outputs/audit"

def read(relative):
    return pd.read_csv(ROOT / relative, sep="\t", dtype=str).fillna("")

def key(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().removeprefix("SUB-"))

def number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (ValueError, TypeError):
        return np.nan

def valid(value, low, high, allowed=None):
    x = number(value)
    if not low <= x <= high:
        return np.nan
    if allowed is not None and x not in allowed:
        return np.nan
    return x

def parse_date(value, epoch=None):
    if value is None or str(value).strip() == "":
        return pd.NaT
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float)):
        if epoch is None or value <= 0:
            return pd.NaT
        try:
            return pd.Timestamp(from_excel(value, epoch)).normalize()
        except Exception:
            return pd.NaT
    return pd.to_datetime(str(value), errors="coerce").normalize()

base = read("inputs/COMMON_subject_measurements_and_membership.tsv")
base = base[base.supported_site.str.lower().isin(["true", "1", "1.0"])].copy()
base["join_id"] = base.subject.map(key)
assert len(base) == 307
assert not base.duplicated(["dataset", "join_id"]).any()

roster = read("inputs/final_DTI_subject_roster_347.tsv")
demo = read("inputs/final_DTI_demographics_347.tsv")
for frame in [roster, demo]:
    frame["join_id"] = frame.subject_bids.map(key)
    assert not frame.duplicated(["dataset", "join_id"]).any()

base = base.merge(
    roster[["dataset", "join_id", "source_subject_id", "adni_rid"]],
    on=["dataset", "join_id"], how="left", validate="one_to_one"
).merge(
    demo[["dataset", "join_id", "selected_DWI_date"]],
    on=["dataset", "join_id"], how="left", validate="one_to_one"
)
base["mri_date"] = base.selected_DWI_date.map(parse_date)
assert base.mri_date.notna().all(), "Missing or invalid MRI dates"
assert base.source_subject_id.notna().all(), "Unmatched roster subjects"

# Candidates retain source record and assessment provenance.
candidates = []
def add(dataset, subject, measure, value, assessment_date, source, record):
    candidates.append(dict(
        dataset=dataset, join_id=key(subject), measure=measure,
        value=value, assessment_date=assessment_date,
        source_file=source, source_record=str(record)
    ))

for filename, fields in [
    ("MMSE_30Sep2026.csv", [("MMSE", "MMSCORE", 0, 30)]),
    ("CDR_15Sep2026.csv", [
        ("CDR_global", "CDGLOBAL", 0, 3),
        ("ADNI_CDRSB", "CDRSB", 0, 18)
    ])
]:
    clinical = pd.read_csv(
        ROOT / "raw/ADNI" / filename, dtype=str
    ).fillna("")
    for index, row in clinical.iterrows():
        if number(row.get("HAS_QC_ERROR")) == 1:
            continue
        if filename.startswith("MMSE") and number(row.get("DONE")) == 0:
            continue
        for measure, field, low, high in fields:
            allowed = {0, .5, 1, 2, 3} if measure == "CDR_global" else None
            value = valid(row.get(field), low, high, allowed)
            if measure == "MMSE" and np.isfinite(value) and value != int(value):
                value = np.nan
            add("ADNI", row["PTID"], measure, value,
                parse_date(row["VISDATE"]), filename,
                row.get("ID") or index + 2)

filename = "NIFD_Clinical_Data_20200724_updated.xlsx"
wb = load_workbook(ROOT / "raw/NIFD" / filename,
                   read_only=True, data_only=True)
iterator = wb.worksheets[0].iter_rows(values_only=True)
headers = next(iterator)
definitions = [
    ("MMSE", "MMSE_TOT", "MMSE_DCDATE", 0, 30),
    ("CDR_global", "CDR_TOT", "CDR_DCDATE", 0, 3),
    ("NIFD_CDR_box", "CDR_BOX_SCORE", "CDR_DCDATE", 0, 24),
    ("CDR_behavior", "CDR_BEHAV", "CDR_DCDATE", 0, 3),
    ("CDR_language", "CDR_LANG", "CDR_DCDATE", 0, 3),
    ("FAQ", "FAQ_TOTAL", "FAQ_DCDATE", 0, 30),
    ("NPI_apathy_presence", "NPI_APA", "NPI_DCDATE", 0, 1),
    ("NPI_disinhibition_presence", "NPI_DISN", "NPI_DCDATE", 0, 1),
]
for index, values in enumerate(iterator, start=2):
    row = dict(zip(headers, values))
    for measure, field, datefield, low, high in definitions:
        allowed = None
        if measure.startswith("CDR") or measure == "CDR_global":
            allowed = {0, .5, 1, 2, 3}
        if measure.endswith("presence"):
            allowed = {0, 1}
        value = valid(row.get(field), low, high, allowed)
        if measure == "MMSE" and np.isfinite(value) and value != int(value):
            value = np.nan
        add("NIFD", row.get("LONI_ID"), measure, value,
            parse_date(row.get(datefield), wb.epoch), filename, index)
    for domain, presence, severity in [
        ("apathy", "NPI_APA", "NPI_APASEV"),
        ("disinhibition", "NPI_DISN", "NPI_DISNSEV")
    ]:
        present = number(row.get(presence))
        value = (0.0 if present == 0 else
                 valid(row.get(severity), 1, 3, {1, 2, 3})
                 if present == 1 else np.nan)
        add("NIFD", row.get("LONI_ID"), f"NPI_{domain}_severity", value,
            parse_date(row.get("NPI_DCDATE"), wb.epoch), filename, index)
wb.close()

candidate = pd.DataFrame(candidates)
candidate = candidate.merge(
    base[["dataset", "join_id", "subject", "group", "mri_date"]],
    on=["dataset", "join_id"], how="inner", validate="many_to_one"
)
candidate["days_from_mri"] = (
    candidate.assessment_date - candidate.mri_date
).dt.days
candidate.to_csv(AUDIT / "clinical_candidates.tsv", sep="\t", index=False)

measures = sorted(candidate.measure.unique())
results = []
for window in [30, 90, 180]:
    for _, participant in base.iterrows():
        rows = candidate[
            (candidate.dataset == participant.dataset) &
            (candidate.join_id == participant.join_id)
        ]
        for measure in measures:
            subset = rows[rows.measure == measure].copy()
            usable = subset[
                subset.value.notna() &
                subset.days_from_mri.notna() &
                (subset.days_from_mri.abs() <= window)
            ].copy()
            result = dict(
                dataset=participant.dataset, subject=participant.subject,
                group=participant.group, measure=measure,
                window_days=window, value=np.nan,
                assessment_date=pd.NaT, days_from_mri=np.nan,
                source_file="", source_record="",
                status="no_valid_score_within_window"
            )
            if subset.empty:
                result["status"] = "measure_not_available_for_dataset"
            if not usable.empty:
                nearest = usable[
                    usable.days_from_mri.abs() ==
                    usable.days_from_mri.abs().min()
                ]
                if nearest.value.nunique() > 1:
                    result["status"] = "conflicting_nearest_scores"
                else:
                    selected = nearest.sort_values(
                        ["assessment_date", "source_record"]
                    ).iloc[0]
                    for field in ["value", "assessment_date", "days_from_mri",
                                  "source_file", "source_record"]:
                        result[field] = selected[field]
                    result["status"] = "matched"
            results.append(result)

matched = pd.DataFrame(results)
matched.to_csv(OUT / "clinical_matches_all_windows.tsv", sep="\t", index=False)
primary = matched[matched.window_days == 90]
wide = primary.pivot(index=["dataset", "subject", "group"],
                     columns="measure", values="value").reset_index()
merged = base.merge(wide, on=["dataset", "subject", "group"],
                    how="left", validate="one_to_one")
merged.to_csv(OUT / "clinical_imaging_307.tsv", sep="\t", index=False)

summary = matched.groupby(
    ["window_days", "dataset", "group", "measure"]
).agg(
    n_total=("subject", "size"), n_available=("value", "count"),
    mean=("value", "mean"), sd=("value", "std"),
    median=("value", "median"),
    q25=("value", lambda x: x.quantile(.25)),
    q75=("value", lambda x: x.quantile(.75)),
    median_abs_gap_days=("days_from_mri", lambda x: x.abs().median()),
    maximum_abs_gap_days=("days_from_mri", lambda x: x.abs().max())
).reset_index()
summary["n_missing"] = summary.n_total - summary.n_available
summary.to_csv(OUT / "clinical_summary_all_windows.tsv", sep="\t", index=False)

print("\nPRIMARY ±90-DAY CLINICAL SUMMARY")
print(summary[
    (summary.window_days == 90) &
    summary.measure.isin(["MMSE", "CDR_global", "ADNI_CDRSB",
                         "NIFD_CDR_box", "CDR_behavior",
                         "NPI_apathy_severity", "FAQ"])
].to_string(index=False))
print("\nUNRESOLVED MATCHES:",
      (primary.status == "conflicting_nearest_scores").sum())
print("\nOutputs:", OUT)
