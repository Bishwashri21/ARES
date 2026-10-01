import re
import string

import numpy as np
import pandas as pd


def create_column_profiles(datasets, profile_type, add_dataset_name=True):
    """Build six normalized structural features for every column in each dataset.

    Each dataset is expected to expose ``dirty_df`` and ``name`` attributes.
    ``profile_type`` is retained in the signature for compatibility with
    callers that select profile-generation modes outside this function.
    """
    profiles = []

    # Compile once because the same checks are applied to every cell.
    alphabetical_pattern = re.compile(r"[A-Za-z]+")
    numerical_pattern = re.compile(r"(([0-9]+)|(([0-9]+)**\\.**([0-9]+)))")

    for dataset in datasets:
        df = dataset.dirty_df
        num_observations = len(df)

        # An empty table cannot be normalized by its row count. Returning an
        # empty profile is clearer than silently producing invalid values.
        if num_observations == 0:
            profiles.append(pd.DataFrame(index=range(1, 7)))
            continue

        column_profiles = []
        for column in df.columns:
            values_as_text = df[column].astype(str)
            value_counts = df[column].value_counts()

            # Count values that occur exactly once; this captures how sparse
            # or identifier-like a column is without retaining the values.
            num_unique_values = len(value_counts[value_counts == 1])
            num_missing_values = df[column].isnull().sum()
            num_alphabetical_values = values_as_text.str.match(
                alphabetical_pattern
            ).sum()
            num_numerical_values = values_as_text.str.match(numerical_pattern).sum()
            num_punctuation_values = values_as_text.isin(string.punctuation).sum()
            num_miscellaneous_values = df[column].nunique()

            column_profile = np.asarray(
                [
                    num_unique_values,
                    num_missing_values,
                    num_alphabetical_values,
                    num_numerical_values,
                    num_punctuation_values,
                    num_miscellaneous_values,
                ],
                dtype=float,
            ) / num_observations
            column_profiles.append(column_profile)

        # Rows correspond to feature types and columns correspond to source
        # attributes, allowing profiles from multiple datasets to be joined.
        profile_df = pd.DataFrame(
            data=np.asarray(column_profiles).T,
            index=range(1, 7),
            columns=df.columns,
        )

        if add_dataset_name:
            # A two-level label avoids name collisions across datasets.
            profile_df.columns = pd.MultiIndex.from_product(
                [[dataset.name], df.columns]
            )

        profiles.append(profile_df)

    return pd.concat(profiles, axis=1) if profiles else pd.DataFrame()
