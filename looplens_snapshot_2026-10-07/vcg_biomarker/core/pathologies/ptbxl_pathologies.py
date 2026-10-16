from __future__ import annotations

from .registry import PathologyBase, PathologyDefinition, PathologyRegistry


class NormalECG(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="NORM",
            description="Normal ECG",
            superclass="NORM",
            subclass="NORM",
            scp_codes=["NORM"],
            references=["Wagner P, et al. PTB-XL: a large publicly available electrocardiography dataset. Sci Data. 2020;7:154."],
        )


class AnteriorMI(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="AMI",
            description="Anterior myocardial infarction",
            superclass="MI",
            subclass="AMI",
            scp_codes=["AMI", "ASMI", "ALMI", "INJAS", "INJAL", "INJLA"],
        )


class InferiorMI(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="IMI",
            description="Inferior myocardial infarction",
            superclass="MI",
            subclass="IMI",
            scp_codes=["IMI", "ILMI", "IPLMI", "IPMI", "INJIN", "INJIL"],
        )


class LateralMI(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="LMI",
            description="Lateral myocardial infarction",
            superclass="MI",
            subclass="LMI",
            scp_codes=["LMI"],
        )


class PosteriorMI(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="PMI",
            description="Posterior myocardial infarction",
            superclass="MI",
            subclass="PMI",
            scp_codes=["PMI"],
        )


class IschemicAnterior(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="ISCA",
            description="Ischemic changes in anterior leads",
            superclass="STTC",
            subclass="ISCA",
            scp_codes=["ISCAL", "ISCAS", "ISCAN", "ISCLA"],
        )


class IschemicInferior(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="ISCI",
            description="Ischemic changes in inferior leads",
            superclass="STTC",
            subclass="ISCI",
            scp_codes=["ISCIN", "ISCIL"],
        )


class IschemicNonspecific(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="ISC_",
            description="Non-specific ischemic changes",
            superclass="STTC",
            subclass="ISC_",
            scp_codes=["ISC_"],
        )


class STTCChanges(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="STTC",
            description="ST-T changes",
            superclass="STTC",
            subclass="STTC",
            scp_codes=["STTC", "NDT", "DIG", "ANEUR", "EL"],
        )


class NonspecificSTChanges(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="NST_",
            description="Non-specific ST changes",
            superclass="STTC",
            subclass="NST_",
            scp_codes=["NST_"],
        )


class LongQT(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="LNGQT",
            description="Long QT interval",
            superclass="STTC",
            subclass="STTC",
            scp_codes=["LNGQT"],
        )


class LAFB_LPFB(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="LAFB/LPFB",
            description="Left anterior/posterior fascicular block",
            superclass="CD",
            subclass="LAFB/LPFB",
            scp_codes=["LAFB", "LPFB"],
        )


class IRBBB(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="IRBBB",
            description="Incomplete right bundle branch block",
            superclass="CD",
            subclass="IRBBB",
            scp_codes=["IRBBB"],
        )


class CRBBB(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="CRBBB",
            description="Complete right bundle branch block",
            superclass="CD",
            subclass="CRBBB",
            scp_codes=["CRBBB"],
        )


class CLBBB(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="CLBBB",
            description="Complete left bundle branch block",
            superclass="CD",
            subclass="CLBBB",
            scp_codes=["CLBBB"],
        )


class AVBlock(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="_AVB",
            description="AV block (1st, 2nd, or 3rd degree)",
            superclass="CD",
            subclass="_AVB",
            scp_codes=["1AVB", "2AVB", "3AVB"],
        )


class IVCB(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="IVCB",
            description="Non-specific intraventricular conduction disturbance",
            superclass="CD",
            subclass="IVCB",
            scp_codes=["IVCB"],
        )


class WPW(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="WPW",
            description="Wolff-Parkinson-White syndrome",
            superclass="CD",
            subclass="WPW",
            scp_codes=["WPW"],
        )


class LVH(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="LVH",
            description="Left ventricular hypertrophy",
            superclass="HYP",
            subclass="LVH",
            scp_codes=["LVH"],
        )


class RVH(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="RVH",
            description="Right ventricular hypertrophy",
            superclass="HYP",
            subclass="RVH",
            scp_codes=["RVH"],
        )


class LAO_LAE(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="LAO/LAE",
            description="Left atrial overload/enlargement",
            superclass="HYP",
            subclass="LAO/LAE",
            scp_codes=["LAO/LAE"],
        )


class RAO_RAE(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="RAO/RAE",
            description="Right atrial overload/enlargement",
            superclass="HYP",
            subclass="RAO/RAE",
            scp_codes=["RAO/RAE"],
        )


class SEHYP(PathologyBase):
    @property
    def definition(self) -> PathologyDefinition:
        return PathologyDefinition(
            name="SEHYP",
            description="Septal hypertrophy",
            superclass="HYP",
            subclass="SEHYP",
            scp_codes=["SEHYP"],
        )


def register_ptbxl_pathologies():
    for cls in [
        NormalECG, AnteriorMI, InferiorMI, LateralMI, PosteriorMI,
        IschemicAnterior, IschemicInferior, IschemicNonspecific,
        STTCChanges, NonspecificSTChanges, LongQT,
        LAFB_LPFB, IRBBB, CRBBB, CLBBB, AVBlock, IVCB, WPW,
        LVH, RVH, LAO_LAE, RAO_RAE, SEHYP,
    ]:
        PathologyRegistry.register(cls())


register_ptbxl_pathologies()
