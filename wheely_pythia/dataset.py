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

    @property
    def columns(self):
        """
        The columns of the :py:class:`pyspark.sql.DataFrame` that have defined
        semantics in this dataset. Note that additional columns may be available
        and will be preserved in the backing dataframe.
        """
        return [
            *self.score_columns,
            *self.spectrum_columns,
            self.target_column,
            self.peptide_column,
            self.protein_column,
            self.charge_column,
            self.mz_column,
            self.rt_column,
            self.peaklist_column,
        ]
