from wheely.mammoth import PsmDataset as _PsmDataset
from wheely.mammoth.spectra.dataset import (
    SpectraDatasetBase as _SpectraDatasetBase,
)


class PythiaDataset(_PsmDataset, _SpectraDatasetBase):
    def __init__(
        self,
        psms,
        target_column,
        score_columns,
        spectrum_columns,
        peptide_column,
        charge_column,
        mz_column,
        rt_column,
        peaklist_column,
        protein_column,
        protein_delim=None,
    ):
        _PsmDataset.__init__(
            self,
            psms,
            target_column,
            score_columns,
            spectrum_columns,
            peptide_column,
            protein_column,
            protein_delim,
        )
        _SpectraDatasetBase.__init__(
            self,
            psms,
            spectrum_columns,
            charge_column,
            mz_column,
            rt_column,
            peaklist_column,
        )
