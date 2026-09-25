import re

import os

import csv

import numpy as np

import pandas as pd

from collections import defaultdict

from sklearn.metrics import mutual_info_score

from itertools import combinations





# ---------------------------------------------------------------------------

# String-pattern aggregation helpers
#
# These functions describe the shape of a value rather than its literal
# contents. Keep the four representations in the same order because the
# feature-construction code below relies on that fixed ordering.

# ---------------------------------------------------------------------------



def L1_str_agg(value):

    pattern = re.compile(r'(\w+)')

    def symbol_replacer(match):

        text = match.group(0)

        if text.isalnum():

            return r'\A-{}'.format(len(text))

        return text

    return pattern.sub(symbol_replacer, value)





def L2_str_agg(s):

    result = []

    current_char_type = None

    count = 0

    for char in s:

        if char.isdigit():

            char_type = 'D'

        elif char.isalpha():

            char_type = 'L'

        else:

            char_type = 'S'

        if char_type != current_char_type:

            if current_char_type is not None:

                result.append(f'\\\\{current_char_type}-{count}')

            current_char_type = char_type

            count = 1

        else:

            count += 1

    if current_char_type is not None:

        result.append(f'\\\\{current_char_type}[{count}]')

    return ''.join(result)





def L3_str_agg(value):

    result = []

    prev_char_type = ""

    count = 0



    def append_l2(char_type, cnt):

        return f"{char_type}-{cnt}" if cnt > 0 else ""



    for char in value:

        if char.isdigit():

            char_type = r"\D"

        elif char.islower():

            char_type = r"\Ll"

        elif char.isupper():

            char_type = r"\Lu"

        elif not char.isalnum():

            char_type = r"\S"

        else:

            continue

        if char_type != prev_char_type and prev_char_type != "":

            result.append(append_l2(prev_char_type, count))

            count = 1

        else:

            count += 1

        prev_char_type = char_type

    result.append(append_l2(prev_char_type, count))

    return ''.join(result)





def str_agg(s):

    """Return [raw, L3, L2, L1] pattern representations of a string value."""

    return [s, L3_str_agg(s), L2_str_agg(s), L1_str_agg(s)]





# ---------------------------------------------------------------------------

# Main feature creator

# ---------------------------------------------------------------------------



def create_zeroed_features(df, top_k_related, fasttext_model_path=None, fasttext_dim=50, k=3):

    """

    Creates ZeroED-style per-column features of fixed size (k + 4 + fasttext_dim*(1+k)):



      - Vicinity  (k features)             : co-occurrence count of (col, val) with each

                                             of the top-k related attribute values.

      - Pattern   (4 features)             : frequency of the L1/L2/L3/raw string

                                             pattern of the value within its column.

      - FastText  (fasttext_dim * (1+k))   : reduced fasttext embedding of the cell

                                             value, followed by embeddings of the values

                                             in each related attribute (zero-padded when

                                             the model is unavailable or fewer than k

                                             related attrs exist).



    The feature width is constant regardless of dataset, so classifiers trained on one

    dataset can be applied to another.



    Args:

        df               : pd.DataFrame  (values will be cast to str)

        top_k_related    : dict {col: [rel_col_1, ..., rel_col_k]}

        fasttext_model_path : path to a fasttext .bin model (optional)

        fasttext_dim     : embedding dimensionality after reduction

        k                : maximum number of related attributes (pad zeros if fewer)



    Returns:

        dict {col_name: pd.DataFrame of shape (n_rows, k + 4 + fasttext_dim*(1+k))}

    """

    # Use one missing-value marker throughout so counting and pattern features
    # treat absent cells consistently.
    df = df.astype(str).fillna('nan')

    all_attrs = list(df.columns)

    col_to_idx = {col: i for i, col in enumerate(all_attrs)}

    n_rows = len(df)

    n_ft_total = fasttext_dim * (1 + k)

    feat_width = k + 4 + n_ft_total



    df_arr = df.values  # shape (n_rows, n_cols), dtype object (strings)



    # --- 1. Co-occurrence counts for vicinity features ---
    # Count only observed pairs. Missing values are not evidence of a
    # relationship between two columns.

    co_occur = defaultdict(lambda: defaultdict(int))

    for row_i in range(n_rows):

        for col_i, col in enumerate(all_attrs):

            val = df_arr[row_i, col_i]

            if val == 'nan':

                continue

            for rel_col in top_k_related.get(col, [])[:k]:

                rel_val = df_arr[row_i, col_to_idx[rel_col]]

                if rel_val != 'nan':

                    co_occur[(col, val)][(rel_col, rel_val)] += 1



    # --- 2. Pattern frequency stats per column ---

    col_pat_stats = {col: {} for col in all_attrs}

    cell_pats = [[None] * len(all_attrs) for _ in range(n_rows)]

    for row_i in range(n_rows):

        for col_i, col in enumerate(all_attrs):

            pats = str_agg(df_arr[row_i, col_i])

            cell_pats[row_i][col_i] = pats

            for pat in pats:

                col_pat_stats[col][pat] = col_pat_stats[col].get(pat, 0) + 1



    # --- 3. Load FastText model (optional) ---
    # Embeddings are optional: zero-filled slots preserve the expected feature
    # width when no compatible model is available.

    ft_model = None

    if fasttext_model_path and os.path.exists(fasttext_model_path):

        try:

            import fasttext

            import fasttext.util

            ft_model = fasttext.load_model(fasttext_model_path)

            if ft_model.get_dimension() > fasttext_dim:

                fasttext.util.reduce_model(ft_model, fasttext_dim)

        except Exception:

            ft_model = None



    # --- 4. Build per-column feature matrices ---
    # Every column receives its own matrix, but all matrices have identical
    # widths so downstream classifiers can use a common input shape.

    col_names = (

        [f"vic_{i}" for i in range(k)] +

        [f"pat_{i}" for i in range(4)] +

        [f"ft_{i}" for i in range(n_ft_total)]

    )



    features = {}

    for col_i, col in enumerate(all_attrs):

        related_cols = top_k_related.get(col, [])[:k]

        related_idx = [col_to_idx[rc] for rc in related_cols]



        mat = np.zeros((n_rows, feat_width), dtype=np.float32)



        for row_i in range(n_rows):

            val = df_arr[row_i, col_i]



            # Vicinity

            for i, (rc, rc_idx) in enumerate(zip(related_cols, related_idx)):

                rel_val = df_arr[row_i, rc_idx]

                mat[row_i, i] = co_occur[(col, val)][(rc, rel_val)]



            # Pattern

            pats = cell_pats[row_i][col_i]

            for p_i, pat in enumerate(pats):

                mat[row_i, k + p_i] = col_pat_stats[col].get(pat, 0)



            # FastText

            if ft_model is not None:

                offset = k + 4

                fv = ft_model.get_word_vector(val)

                mat[row_i, offset:offset + fasttext_dim] = fv[:fasttext_dim]

                for j, (rc, rc_idx) in enumerate(zip(related_cols, related_idx)):

                    rel_val = df_arr[row_i, rc_idx]

                    rfv = ft_model.get_word_vector(rel_val)

                    ro = offset + (j + 1) * fasttext_dim

                    mat[row_i, ro:ro + fasttext_dim] = rfv[:fasttext_dim]



        features[col] = pd.DataFrame(mat, columns=col_names)



    return features



def execute_func(function_code, val, attr):
    # Criteria are supplied as Python source. A failure is treated as an unmet
    # criterion so one invalid rule does not stop feature generation.

    local_scope = {}

    try:

        exec(function_code, globals(), local_scope)

        function_name = list(local_scope.keys())[0]

        function = local_scope[function_name]

        return function(val, attr)

    except Exception:

        return False



def handle_func_exec(func, val, attr):

    try:

        result = execute_func(func, val, attr)

    except Exception:

        return -1

    return 1 if result else 0



def create_criteria_features(df, criteria_functions):

    """

    Generates features based on LLM-generated criteria functions.

    criteria_functions: dict {col_name: [func_code1, func_code2, ...]}

    """

    features = {}

    for col in df.columns:

        col_funcs = criteria_functions.get(col, [])

        if not col_funcs:

            continue

        col_features = []

        for func_code in col_funcs:

            # Apply each criterion independently; one output column is created
            # for every supplied rule.

            res = df[col].apply(lambda x: handle_func_exec(func_code, x, col))

            col_features.append(res)

        if col_features:

            features[col] = pd.concat(col_features, axis=1)

            features[col].columns = [f"criteria_{i}" for i in range(len(col_funcs))]

        else:

            features[col] = pd.DataFrame()

    return features



# Aggregated features (NMI based)



def cal_mutual_information(col1, col2):
    # Normalize missing and non-finite values before comparing discrete values.

    col1 = col1.replace([np.nan, None, np.inf, -np.inf, 'nan'], 'NA').astype(str)

    col2 = col2.replace([np.nan, None, np.inf, -np.inf, 'nan'], 'NA').astype(str)

    mask = (col1 != 'NA') & (col2 != 'NA')

    col1, col2 = col1[mask], col2[mask]

    if len(col1) == 0 or len(col2) == 0:

        return 0.0

    return mutual_info_score(col1, col2)



def cal_entropy(column):

    # Ensure column is string and handle NaNs uniformly before unique

    column = column.astype(str)

    column = column[column != 'nan']

    if len(column) == 0:

        return 0.0

    _, counts = np.unique(column, return_counts=True)

    probabilities = counts / len(column)

    return -sum(p * np.log2(p) for p in probabilities if p > 0)



def cal_nmi(column1, column2):

    mi = cal_mutual_information(column1, column2)

    entropy1 = cal_entropy(column1)

    entropy2 = cal_entropy(column2)

    if entropy1 == 0 and entropy2 == 0: return 1.0

    if entropy1 == 0 or entropy2 == 0: return 0.0

    return 2 * mi / (entropy1 + entropy2)



def get_top_k_related_attributes(df, k=3):
    # Compute each unordered pair once, then store the score in both directions
    # because every column needs its own related-attribute ranking.

    columns = df.columns

    nmi_results = {}

    for col1, col2 in combinations(columns, 2):

        nmi = cal_nmi(df[col1], df[col2])

        nmi_results[(col1, col2)] = nmi

        nmi_results[(col2, col1)] = nmi

    top_k = defaultdict(list)

    for col in columns:

        related = []

        for other_col in columns:

            if col == other_col: continue

            nmi = nmi_results.get((col, other_col), 0)

            related.append((other_col, nmi))

        related.sort(key=lambda x: x[1], reverse=True)

        top_k[col] = [x[0] for x in related[:k]]

    return top_k



def create_aggregated_features(df, top_k_related):

    """

    Create one numeric co-occurrence feature for each related attribute.

    A row's feature value is the number of rows that contain the same pair of
    values. This preserves local cross-column context without placing raw
    categorical values directly in the feature vector.

    """

    # 1. Calculate co-occurrence counts

    # This can be expensive, so compute all pair counts once and reuse them.

    features = {}

    # Pre-compute co-occurrences

    co_occur_counts = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():

        for col in df.columns:

            val = row[col]

            for rel_col in top_k_related.get(col, []):

                rel_val = row[rel_col]

                co_occur_counts[(col, val)][(rel_col, rel_val)] += 1

    for col in df.columns:

        col_feats = []

        related_cols = top_k_related.get(col, [])

        for rel_col in related_cols:

            # Feature: Count of co-occurrence of (col_val, rel_col_val)

            feat = []

            for idx, row in df.iterrows():

                val = row[col]

                rel_val = row[rel_col]

                count = co_occur_counts[(col, val)][(rel_col, rel_val)]

                feat.append(count)

            col_feats.append(pd.Series(feat, name=f"co_occur_{rel_col}"))

        if col_feats:

            features[col] = pd.concat(col_feats, axis=1)

        else:

            features[col] = pd.DataFrame()

    return features
