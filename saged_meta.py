"""Column-similarity matching and model-based meta-feature generation."""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

from zeroed_features import create_zeroed_features, get_top_k_related_attributes
from saged_profiler import create_column_profiles


def get_similarity(dirty_dataset, historical_datasets, config, verbose=False):
    """Match every dirty column to historical columns in the same profile cluster.

    Dataset objects must expose ``name`` and ``dirty_df``.  Historical-column
    identifiers are returned as ``(dataset_name, column_name)`` tuples.
    """
    # Profiles describe column structure, so columns are the observations given
    # to KMeans.  The dirty columns are placed first to keep their positions
    # easy to identify after clustering.
    dirty_profiles = create_column_profiles(
        [dirty_dataset], config.profile_type, add_dataset_name=False
    )
    historical_profiles = create_column_profiles(
        historical_datasets, config.profile_type, add_dataset_name=True
    )
    all_profiles = pd.concat([dirty_profiles, historical_profiles], axis=1)

    if all_profiles.shape[1] == 0:
        return defaultdict(list)
    if config.n_clusters > all_profiles.shape[1]:
        raise ValueError("n_clusters cannot exceed the number of profiled columns.")

    kmeans = KMeans(n_clusters=config.n_clusters)
    labels = kmeans.fit_predict(all_profiles.T)

    similarity = defaultdict(list)
    dirty_columns = dirty_dataset.dirty_df.columns
    historical_start = len(dirty_columns)

    for dirty_index, dirty_column in enumerate(dirty_columns):
        dirty_cluster = labels[dirty_index]

        # Historical profiles use a two-level column label: dataset name first,
        # then the source column name.
        for profile_index in range(historical_start, len(labels)):
            if labels[profile_index] == dirty_cluster:
                historical_name, historical_column = all_profiles.columns[profile_index]
                similarity[dirty_column].append((historical_name, historical_column))

    return similarity


def create_meta_features(dirty_dataset, historical_datasets, config, verbose=False):
    """Prepare column similarity until historical base models are supplied.

    Use :func:`train_base_classifiers` followed by
    :func:`generate_meta_features_with_models` for the full prediction-based
    pipeline. This compatibility wrapper retains the original empty dataframe
    output while ensuring that column matching is performed consistently.
    """
    get_similarity(dirty_dataset, historical_datasets, config, verbose)
    return pd.DataFrame()


def train_base_classifiers(historical_datasets, fasttext_model_path=None, fasttext_dim=50, k=3):
    """Train one error-probability MLP for every eligible historical column."""
    models = {}

    for dataset in historical_datasets:
        top_k_related = get_top_k_related_attributes(dataset.dirty_df, k=k)
        features = create_zeroed_features(
            dataset.dirty_df,
            top_k_related,
            fasttext_model_path=fasttext_model_path,
            fasttext_dim=fasttext_dim,
            k=k,
        )
        errors = dataset.get_actual_errors()

        for column in dataset.dirty_df.columns:
            if column not in features or column not in errors:
                continue

            # A probability classifier requires both clean and erroneous
            # examples.  Constant-label columns carry no trainable signal.
            target = errors[column]
            if target.nunique() < 2:
                continue

            classifier = MLPClassifier(max_iter=500)
            classifier.fit(features[column].fillna(0), target)
            models[(dataset.name, column)] = classifier

    return models


def generate_meta_features_with_models(
    dirty_dataset, models, similarity, config, dirty_features=None
):
    """Apply matching historical classifiers and summarize their predictions.

    The four output values per row are the mean, maximum, minimum, and standard
    deviation of all available historical error probabilities for that column.
    """
    if dirty_features is None:
        top_k_related = get_top_k_related_attributes(dirty_dataset.dirty_df)
        dirty_features = create_zeroed_features(
            dirty_dataset.dirty_df,
            top_k_related,
            fasttext_model_path=getattr(config, "fasttext_model_path", None),
            fasttext_dim=getattr(config, "fasttext_dim", 50),
        )

    meta_features = {}
    zero_columns = ["meta_mean", "meta_max", "meta_min", "meta_std"]

    for column in dirty_dataset.dirty_df.columns:
        predictions = []

        for historical_dataset_name, historical_column in similarity.get(column, []):
            model = models.get((historical_dataset_name, historical_column))
            if model is None:
                continue

            # Use raw arrays because feature labels differ between the target
            # column and the historical column even when their widths match.
            input_features = dirty_features[column].fillna(0).to_numpy()
            predictions.append(model.predict_proba(input_features)[:, 1])

        if predictions:
            prediction_array = np.asarray(predictions)
            meta_features[column] = pd.DataFrame(
                {
                    "meta_mean": prediction_array.mean(axis=0),
                    "meta_max": prediction_array.max(axis=0),
                    "meta_min": prediction_array.min(axis=0),
                    "meta_std": prediction_array.std(axis=0),
                }
            )
        else:
            # Preserve the same feature schema when no historical model was
            # available for a target column.
            meta_features[column] = pd.DataFrame(
                0.0,
                index=dirty_dataset.dirty_df.index,
                columns=zero_columns,
            )

    return meta_features
