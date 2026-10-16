import pandas as pd
import pytest

from vcg_biomarker.mimic.code15_cohort import read_exam_metadata


def test_only_explicit_sami_contract_can_derive_patient_identity(tmp_path):
    path = tmp_path / "exams.csv"
    pd.DataFrame({"exam_id": [1, 2]}).to_csv(path, index=False)
    for cohort in (None, "code15"):
        with pytest.raises(ValueError, match="Missing patient_id"):
            read_exam_metadata(path, cohort=cohort)
    output = read_exam_metadata(path, cohort="sami_trop")
    assert output.patient_id.tolist() == [1, 2]
    assert output.patient_identity_namespace.eq("sami_trop").all()
    assert output.patient_id_source.eq("derived_one_exam_per_participant").all()


@pytest.mark.parametrize("ids", [[1, 1], [1, None], [1, 1.5], [1, 0]])
def test_invalid_exam_keys_rejected(tmp_path, ids):
    path = tmp_path / "exams.csv"
    pd.DataFrame({"exam_id": ids, "patient_id": [10, 10]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Exam identifiers"):
        read_exam_metadata(path, cohort="code15")


def test_repeated_patient_preserved_missing_patient_rejected(tmp_path):
    path = tmp_path / "exams.csv"
    pd.DataFrame({"exam_id": [1, 2], "patient_id": [10, 10]}).to_csv(path, index=False)
    assert read_exam_metadata(path, cohort="code15").patient_id.nunique() == 1
    pd.DataFrame({"exam_id": [1, 2], "patient_id": [10, None]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Patient identifiers"):
        read_exam_metadata(path, cohort="code15")
