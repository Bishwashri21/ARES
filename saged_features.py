import string

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class Word2VecFeatures:
    """Create one Word2Vec embedding block for each column of a table."""

    def __init__(self, vector_size=100, epochs=10):
        self.vector_size = vector_size
        self.epochs = epochs

    def add_word2vec_features(
        self, dataset, all_matrix_train, feature_name_list, word2vec_only=False
    ):
        """Fit embeddings, then append them to an existing feature matrix."""
        self.fit(dataset)
        features_train = self.transform(dataset)
        names, _ = self.get_feature_names(dataset)

        if word2vec_only:
            all_features_train_new = features_train
            feature_name_list = names
        else:
            # The first feature block has no previous matrix to concatenate.
            if all_matrix_train is None or (
                hasattr(all_matrix_train, "shape") and all_matrix_train.shape[0] == 0
            ):
                all_features_train_new = features_train
            else:
                all_features_train_new = hstack(
                    (all_matrix_train, features_train)
                ).tocsr()
            feature_name_list.extend(names)

        return all_features_train_new, feature_name_list, None

    def get_feature_names(self, dataset):
        """Return stable names and per-column embedding widths."""
        names = []
        feature_number_per_col = {}

        for column_index, column_name in enumerate(dataset.columns):
            column_names = [
                f"{column_name}_word2vec_{vector_index}"
                for vector_index in range(self.vector_size)
            ]
            names.extend(column_names)
            feature_number_per_col[column_index] = len(column_names)

        return names, feature_number_per_col

    def fit(self, data):
        """Map each column value to a column-specific token and train Word2Vec."""
        if data.empty:
            raise ValueError("Word2Vec features require at least one data row.")

        self.column_dictionaries = []
        words = np.empty((data.shape[0], data.shape[1]), dtype=object)

        for column_index in range(data.shape[1]):
            value_to_word = {}
            for row_index in range(data.shape[0]):
                value = data.iloc[row_index, column_index]
                if value not in value_to_word:
                    # Prefixing with the column index prevents equal values in
                    # different attributes from being treated as one token.
                    value_to_word[value] = f"col{column_index}_{len(value_to_word)}"
                words[row_index, column_index] = value_to_word[value]
            self.column_dictionaries.append(value_to_word)

        # Each row is one short context sequence containing its column tokens.
        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=words.shape[1] * 2,
            min_count=1,
            workers=4,
            negative=0,
            hs=1,
        )
        # Build the vocabulary separately so the requested epoch count is the
        # exact number of training passes, rather than an additional pass.
        self.model.build_vocab(words.tolist())
        self.model.train(
            words.tolist(), total_examples=words.shape[0], epochs=self.epochs
        )

    def transform(self, data):
        """Convert each row into the concatenated embeddings of its cell values."""
        words = np.empty((data.shape[0], data.shape[1]), dtype=object)

        for column_index in range(data.shape[1]):
            value_to_word = self.column_dictionaries[column_index]
            for row_index in range(data.shape[0]):
                value = data.iloc[row_index, column_index]
                # Unseen values receive an all-zero embedding below.
                words[row_index, column_index] = value_to_word.get(value, "unknown")

        features = []
        for row in words:
            row_features = []
            for word in row:
                if word in self.model.wv:
                    row_features.extend(self.model.wv[word])
                else:
                    row_features.extend(np.zeros(self.vector_size))
            features.append(row_features)

        return np.asarray(features)


def create_tf_idf(column, chars=None):
    """Create character-level TF-IDF features, padded to a fixed character set."""
    if chars is None:
        chars = list(string.printable)

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(analyzer="char", lowercase=False, ngram_range=(1, 1))),
            ("tfidf", TfidfTransformer()),
        ]
    )

    if column.empty:
        return pd.DataFrame(columns=chars, dtype=float)

    tf_idf = pipeline.fit_transform(column.fillna("").astype(str))
    observed_features = pd.DataFrame(
        tf_idf.toarray(), columns=pipeline.get_feature_names_out()
    )

    # Keep a predictable set of character columns even when a character is
    # absent from the current dataset.
    padded_features = pd.DataFrame(np.zeros((len(column), len(chars))), columns=chars)
    padded_features.update(observed_features)
    return padded_features


def create_w2v(df, vector_size=100, epochs=10):
    """Fit table-wide embeddings and return them as a labeled dataframe."""
    model = Word2VecFeatures(vector_size=vector_size, epochs=epochs)
    features, feature_names, _ = model.add_word2vec_features(df, None, [])
    return pd.DataFrame(features, columns=feature_names)


def create_metadata(column):
    """Create frequency, type, length, alphabetic, and numeric-value features."""
    value_counts = column.value_counts(dropna=False)

    def is_number(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    # ``reindex`` is reliable for repeated values and missing values alike.
    occurrence_counts = value_counts.reindex(column.to_numpy()).to_numpy()

    return pd.DataFrame(
        np.asarray(
            [
                occurrence_counts,
                column.map(lambda value: int(is_number(value))),
                column.astype(str).map(len),
                column.astype(str).map(str.isalpha).astype(int),
                column.map(
                    lambda value: float(value)
                    if is_number(value) and not pd.isna(value)
                    else 0
                ),
            ]
        ).T,
        columns=[
            "num_occurrences",
            "is_numeric",
            "length",
            "is_alphabetical",
            "extracted_number",
        ],
    )


def create_features(df, tf_idf_chars=None, w2v_size=100):
    """Build a combined feature dataframe for each column in ``df``."""
    if df.empty:
        return {column: pd.DataFrame(index=df.index) for column in df.columns}

    features = {}

    # Word2Vec is trained once on the complete table to capture cross-column
    # co-occurrence; each column then selects its own embedding block.
    w2v_features = create_w2v(df, vector_size=w2v_size)

    for column in df.columns:
        tf_idf = create_tf_idf(df[column], chars=tf_idf_chars)
        metadata = create_metadata(df[column])
        column_w2v_columns = [
            name
            for name in w2v_features.columns
            if name.startswith(f"{column}_word2vec_")
        ]
        column_w2v = w2v_features[column_w2v_columns].reset_index(drop=True)

        features[column] = pd.concat([tf_idf, metadata, column_w2v], axis=1)

    return features
