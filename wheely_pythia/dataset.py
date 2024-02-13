from wheely.mammoth import PsmDataset as _PsmDataset
from wheely.mammoth.spectra.dataset import (
    PrecursorDatasetBase as _PrecursorDatasetBase,
    SpectraDatasetMixin as _SpectraDatasetMixin,
)


class PythiaDataset(_PsmDataset, _PrecursorDatasetBase):
    def __init__(
        self,
        psms,
        target_column,
        score_columns,
        spectrum_columns,
        charge_column,
        rt_column,
        peptide_column,
        protein_column,
        protein_delim=None,
        mz_column="mz",
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
        _PrecursorDatasetBase.__init__(
            self,
            psms,
            spectrum_columns,
            charge_column,
            mz_column,
            rt_column,
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
        ]


class PythiaSpectraDataset(PythiaDataset, _SpectraDatasetMixin):
    def __init__(
        self,
        psms,
        target_column,
        score_columns,
        spectrum_columns,
        charge_column,
        rt_column,
        peptide_column,
        protein_column,
        protein_delim=None,
        mz_column="mz",
        peaklist_column="peaklist",
    ):
        PythiaDataset.__init__(
            self,
            psms=psms,
            target_column=target_column,
            score_columns=score_columns,
            spectrum_columns=spectrum_columns,
            rt_column=rt_column,
            peptide_column=peptide_column,
            protein_column=protein_column,
            protein_delim=protein_delim,
            charge_column=charge_column,
            mz_column=mz_column,
        )
        _SpectraDatasetMixin.__init__(
            self,
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
