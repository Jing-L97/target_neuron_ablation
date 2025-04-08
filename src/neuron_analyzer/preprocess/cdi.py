# ruff: noqa
# type: ignore
"""
Loader and extractor for Human CDI.

!!! Note that this script is migrated from another repo and haven't integreated well
"""

import enum
import functools
from pathlib import Path

import pandas as pd
import spacy

from lexical_benchmark import settings
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils import text_cleaning


# TODO: fit into our package
class POSTypes(str, enum.Enum):
    """Categories for Part of speech (PoS)."""

    all = "all"
    content = "content"
    function = "function"

    def __str__(self) -> str:
        """As string."""
        return self.value


class CDIPreparation:
    """Prepare the MB-CDI csv.

    To Download CDI data you have to use the wordbank website.
    URL: https://wordbank.stanford.edu/data?name=item_data
    This CSV contains ... TODO add brief description
    More info: docs/datasets/human_cdi (TODO: add correct documentation file)

    Column definitions:
        - downloaded: a date corresponding to the date it was downloaded
        - item_id: the id of the utterance
        - item_definition: the definition of the utterance
        - category: the category of the utterance
        - [16 - 30] these columns correspond to the month of the child
            - the cell values correspond to score based on knowledge of the word

    Returns
    -------
    CDI<pd.DataFrame>:
        - word<str>: (the word)
        - word_length<int>: the length of the utterance in characters
        - POS<str>: a string representing the part of speech the utterance belongs to (see docs/... TODO)
        - category: string representing the category of the utterance
        - [16 - 30] these columns correspond to the month of the child
            - the cell values correspond to score based on knowledge of the word

    """

    @property
    def pos_model_load(self) -> spacy.Language:
        """Load POS model from spacy."""
        return dataset_utils.spacy_model("en_core_web_trf")

    def __init__(
        self,
        age_min: int,
        age_max: int,
        raw_csv: Path,
        pos_filter_type: POSTypes = POSTypes.content,
    ) -> None:
        if not raw_csv.is_file():
            raise ValueError(f"Given file ::{raw_csv}:: does not exist !!")

        self._raw_csv_location = raw_csv
        self.pos_filter_type = pos_filter_type
        self.age_min = age_min
        self.age_max = age_max

        # Zero init
        self.download_date = None
        self._src_df: pd.DataFrame | None = None
        self._target_df: pd.DataFrame | None = None
        self.columns = ["word", "category", *[str(a) for a in range(age_min, age_max + 1)]]

    @property
    def df(self) -> pd.DataFrame:
        """Get the data as a Pandas DataFrame."""
        if self._src_df is None:
            self._src_df = self.load_dataset()
        return self._src_df

    @property
    def gold(self) -> pd.DataFrame:
        """Get the Gold data as a Pandas DataFrame."""
        if self._target_df is None:
            self._target_df = self.build_gold()
        return self._target_df

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset into a dataframe."""
        df = pd.read_csv(self._raw_csv_location)
        self.dl_date = df["downloaded"].iloc[0]
        return df.drop(["downloaded"], axis=1)

    def add_POS(
        self,
        df: pd.DataFrame,
        *,
        do_type_filtering: bool = True,
    ) -> pd.DataFrame:
        """Add POS tag to the dataframe."""
        # Load POS inference model and inject it into the word_to_pos function
        word_to_pos = functools.partial(dataset_utils.word_to_pos, pos_model=self.pos_model_load)

        # Create a column POS using previous function
        df["POS"] = df["word"].apply(word_to_pos)

        if do_type_filtering:
            # Filter words by PoS
            if self.pos_filter_type == POSTypes.content:
                # filter out all PoS that is not in CONTENT_POS
                df = df[df["POS"].isin(settings.CONTENT_POS)]
            elif self.pos_filter_type == POSTypes.function:
                # Filter out all PoS that is in CONTENT_POS
                df = df[~df["POS"].isin(settings.CONTENT_POS)]
        return df

    def build_gold(
        self,
        *,
        generate_pos: bool = False,
        do_type_filtering: bool = False,
        filter_categories: bool = True,
        filter_item_definitions: bool = True,
    ) -> pd.DataFrame:
        """Build the gold dataframe from the given src."""
        df = self.df.copy()
        columns = self.columns

        # Explode multidefinition words, clean item definition
        df = dataset_utils.segment_synonym(df, "item_definition")

        # Clean words (normalise accents and non-printable characters)
        word_cleaning = text_cleaning.AZFilter(clean_diacritics=True)
        df["word"] = df["item_definition"].apply(word_cleaning)

        # remove expressions (multi-word lines)
        df = dataset_utils.remove_exp(df, "word")

        # Calculate Word length
        df["word_length"] = df["word"].apply(len)
        columns.append("word_length")

        # POS
        if generate_pos:
            columns.append("POS")
            df = self.add_POS(df, do_type_filtering=do_type_filtering)

        # Filter polysemous words by annotations from original data
        if filter_categories:
            df = df[~df["category"].isin(settings.CATEGORY)]

        if filter_item_definitions:
            df = df[~df["item_definition"].isin(settings.WORD)]

        # merge different word senses by adding the prop
        df = dataset_utils.merge_word(df, "word")
        df = df.drop(["item_id"], axis=1)

        return df[columns].copy()
