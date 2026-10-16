"""Batched note parsing must equal the former whole-frame read, row for row."""
from __future__ import annotations

import pandas as pd

from vcg_biomarker.mimic import note_labels
from vcg_biomarker.tests.test_provenance_integrity import _config


def _write_inputs(config) -> None:
    root = config.standardized_root
    root.mkdir(parents=True, exist_ok=True)
    diagnoses = [
        "Discharge Diagnosis:\nAtrial fibrillation.\n",
        "Discharge Diagnosis:\nNo atrial fibrillation. Heart failure.\n",
        "Discharge Diagnosis:\nPossible myocardial infarction.\n",
        "Discharge Diagnosis:\nPneumonia.\n",
    ]
    discharge = pd.DataFrame({
        "subject_id": [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6],
        # Nulls inside and across batch boundaries, a repeated admission
        # whose states must merge in file order.
        "hadm_id": [10, 10, None, 20, 30, None, 40, 50, 50, None, 60],
        "text": [diagnoses[i % 4] for i in range(11)],
    })
    discharge.to_parquet(root / "discharge.parquet", index=False, row_group_size=3)
    notes = [f"n{i}" for i in range(9)]
    impressions = [
        "IMPRESSION: Moderate pulmonary edema. No pneumothorax.",
        "IMPRESSION: Small pleural effusion.",
        "IMPRESSION: No acute intracranial hemorrhage.",
        "IMPRESSION: No pulmonary embolism.",
    ]
    radiology = pd.DataFrame({
        "note_id": notes,
        "subject_id": [1, 2, 3, 4, 5, 6, 1, 2, 3],
        "hadm_id": [10, 20, None, 40, 50, 60, 10, None, 30],
        "text": [impressions[i % 4] for i in range(9)],
    })
    radiology.to_parquet(root / "radiology.parquet", index=False, row_group_size=2)
    exam = ["CHEST (PA AND LAT)", "CT HEAD W/O CONTRAST", "CTA CHEST", "CT CHEST W/CONTRAST",
            "CHEST PORT. LINE PLACEMENT", "MR HEAD", "CHEST FLUORO", "CHEST (PORTABLE AP)", "CTA CHEST"]
    detail = pd.DataFrame({
        "note_id": notes + notes,
        "field_name": ["exam_name"] * 9 + ["exam_code"] * 9,
        # The non-exam field repeats chest words: it must never select a note.
        "field_value": exam + ["CHEST"] * 9,
    })
    detail.to_parquet(root / "radiology_detail.parquet", index=False, row_group_size=4)


def _run(tmp_path, batch_rows, monkeypatch):
    config = _config(tmp_path)
    _write_inputs(config)
    monkeypatch.setattr(note_labels, "NOTE_BATCH_ROWS", batch_rows)
    manifest = note_labels.build_note_labels(config, force=True)
    flags = pd.read_parquet(config.evidence_root / "note_encounter_flags.parquet")
    mentions = pd.read_parquet(config.evidence_root / "note_mentions.parquet")
    return manifest, flags, mentions


def test_batched_note_parse_equals_single_batch(tmp_path, monkeypatch):
    one, flags_one, mentions_one = _run(tmp_path / "one", 10**6, monkeypatch)
    small, flags_small, mentions_small = _run(tmp_path / "small", 2, monkeypatch)
    pd.testing.assert_frame_equal(flags_one, flags_small)
    pd.testing.assert_frame_equal(mentions_one, mentions_small)
    assert one.row_counts == small.row_counts
    # Rows without an admission are ignored exactly as before; every other
    # admission is present, including the merged repeat (subject 1, hadm 10).
    assert set(zip(flags_one.subject_id, flags_one.hadm_id)) == {
        (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)}
    kinds = mentions_one.groupby("note_kind").size().to_dict()
    assert kinds.get("radiology_chest_xray", 0) > 0 and kinds.get("discharge", 0) > 0
    # Only genuine plain chest films reach the chest-X-ray parser: the CT
    # chest (n3) and fluoroscopy (n6) notes never do.
    chest = mentions_one[mentions_one.note_kind == "radiology_chest_xray"]
    assert set(zip(chest.subject_id, chest.hadm_id)) <= {(1, 10), (5, 50)}
