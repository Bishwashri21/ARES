"""Create controlled, cell-level corruptions for tabular benchmarks.

The injector learns lightweight patterns from a clean table and uses them to
introduce plausible errors. Each successful change is recorded so the clean
and corrupted tables can later provide cell-level evaluation labels. These
heuristics are dataset-agnostic and are not substitutes for domain rules.
"""

import pandas as pd

import numpy as np

import random

import re

from typing import Dict, List, Union, Tuple

from scipy.stats import chi2_contingency, pearsonr

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')



# Butterfinger function for keyboard-based corruptions

def butterfinger(text, prob=0.6, keyboard='querty'):

    """Replace characters with nearby keyboard keys to mimic ordinary typos."""

    keyApprox = {}

    if keyboard == "querty":

        keyApprox['q'] = "qwasedzx"

        keyApprox['w'] = "wqesadrfcx"

        keyApprox['e'] = "ewrsfdqazxcvgt"

        keyApprox['r'] = "retdgfwsxcvgt"

        keyApprox['t'] = "tryfhgedcvbnju"

        keyApprox['y'] = "ytugjhrfvbnji"

        keyApprox['u'] = "uyihkjtgbnmlo"

        keyApprox['i'] = "iuojlkyhnmlp"

        keyApprox['o'] = "oipklujm"

        keyApprox['p'] = "plo['ik"

        keyApprox['a'] = "aqszwxwdce"

        keyApprox['s'] = "swxadrfv"

        keyApprox['d'] = "decsfaqgbv"

        keyApprox['f'] = "fdgrvwsxyhn"

        keyApprox['g'] = "gtbfhedcyjn"

        keyApprox['h'] = "hyngjfrvkim"

        keyApprox['j'] = "jhknugtblom"

        keyApprox['k'] = "kjlinyhn"

        keyApprox['l'] = "lokmpujn"

        keyApprox['z'] = "zaxsvde"

        keyApprox['x'] = "xzcsdbvfrewq"

        keyApprox['c'] = "cxvdfzswergb"

        keyApprox['v'] = "vcfbgxdertyn"

        keyApprox['b'] = "bvnghcftyun"

        keyApprox['n'] = "nbmhjvgtuik"

        keyApprox['m'] = "mnkjloik"

        keyApprox[' '] = " "

    else:

        print ("Keyboard not supported.")

    # The original routine samples from 100 outcomes for every character.
    probOfTypo = int(prob * 100)

    buttertext = ""

    for letter in text:

        lcletter = letter.lower()

        if not lcletter in keyApprox.keys():

            newletter = lcletter

        else:

            if random.choice(range(0, 100)) <= probOfTypo:

                newletter = random.choice(keyApprox[lcletter])

            else:

                newletter = lcletter

        # go back to original case

        if not lcletter == letter:

            newletter = newletter.upper()

        buttertext += newletter

    return buttertext



class UniversalErrorInjector:

    def __init__(self, seed=42):

        """

        Universal Domain-Agnostic Error Injector

        Automatically discovers data patterns and introduces realistic errors



        Args:

            seed (int): Random seed for reproducibility

        """

        np.random.seed(seed)

        random.seed(seed)



        # Character substitution patterns (OCR, keyboard, visual similarity)

        self.char_substitutions = {

            # OCR-like errors

            'o': ['0', 'ο', 'О'], 'O': ['0', 'Ο', 'О'],

            'l': ['1', 'I', 'ǀ', '|'], 'L': ['1', 'I', 'ǀ'],

            'i': ['1', 'l', 'ǀ', '!'], 'I': ['1', 'l', 'ǀ', '!'],

            # Keyboard proximity errors

            'a': ['s', 'q', 'w'], 's': ['a', 'd', 'w'], 'd': ['s', 'f', 'e'],

            'q': ['w', 'a', '1'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],

            # Visual similarity

            'rn': ['m'], 'm': ['rn'], 'vv': ['w'], 'cl': ['d'],

            # Numbers

            '1': ['l', 'I'], '0': ['O', 'o'], '5': ['S'], '8': ['B']

        }



        # Pattern templates for different data types

        self.pattern_templates = {}



    def analyze_data_structure(self, df: pd.DataFrame) -> Dict:

        """

        Automatically analyze data structure and discover patterns

        """

        analysis = {

            'column_types': {},

            'patterns': {},

            'relationships': {},

            'distributions': {},

            'constraints': {}

        }



        # Profile columns first; cross-column relationships are easier to
        # interpret once their individual types and distributions are known.
        for col in df.columns:

            # Detect column type and patterns

            analysis['column_types'][col] = self._detect_column_type(df[col])

            analysis['patterns'][col] = self._extract_patterns(df[col])

            analysis['distributions'][col] = self._analyze_distribution(df[col])

            analysis['constraints'][col] = self._detect_constraints(df[col])



        # Find relationships between columns

        analysis['relationships'] = self._find_relationships(df)



        return analysis



    def _detect_column_type(self, series: pd.Series) -> str:

        """Detect the semantic type of a column"""

        non_null = series.dropna().astype(str)

        if len(non_null) == 0:

            return 'empty'



        samples = non_null.head(100).str.lower()



        # Check for common patterns

        if samples.str.match(r'^\d+$').all():

            return 'integer_id'

        elif samples.str.match(r'^[+-]?(?:\d+\.\d+|\.\d+)$').mean() > 0.8:

            return 'float'

        elif samples.str.contains(r'[:/]').mean() > 0.5:

            return 'time_or_ratio'

        elif samples.str.match(r'^[a-z]{2,4}$').mean() > 0.8:

            return 'code'

        elif samples.str.contains(r'[@.]').mean() > 0.5:

            return 'email_or_url'

        elif samples.str.len().mean() > 50:

            return 'long_text'

        elif samples.str.contains(r'[0-9]').mean() > 0.7:

            return 'mixed_alphanumeric'

        elif samples.str.match(r'^[a-zA-Z\s]+$').mean() > 0.8:

            return 'text'

        else:

            return 'mixed'



    def _extract_patterns(self, series: pd.Series) -> Dict:

        """Extract common patterns in the column"""

        non_null = series.dropna().astype(str)

        if len(non_null) == 0:

            return {}



        patterns = {}



        # Length patterns

        lengths = non_null.str.len()

        patterns['length_mean'] = lengths.mean()

        patterns['length_std'] = lengths.std()

        patterns['length_mode'] = lengths.mode().iloc[0] if len(lengths.mode()) > 0 else None



        # Character patterns

        patterns['has_digits'] = non_null.str.contains(r'\d').mean()

        patterns['has_letters'] = non_null.str.contains(r'[a-zA-Z]').mean()

        patterns['has_special'] = non_null.str.contains(r'[^a-zA-Z0-9\s]').mean()

        patterns['has_spaces'] = non_null.str.contains(r'\s').mean()



        # Common separators

        separators = ['-', '_', '.', '/', ':', ',', ';']

        for sep in separators:

            patterns[f'has_{sep}'] = non_null.str.contains(re.escape(sep)).mean()



        # Case patterns

        patterns['all_upper'] = non_null.str.isupper().mean()

        patterns['all_lower'] = non_null.str.islower().mean()

        patterns['title_case'] = non_null.str.istitle().mean()



        return patterns



    def _analyze_distribution(self, series: pd.Series) -> Dict:

        """Analyze the statistical distribution of the column"""

        distribution = {}



        if pd.api.types.is_numeric_dtype(series):

            clean_series = pd.to_numeric(series, errors='coerce').dropna()

            if len(clean_series) > 0:

                distribution['min'] = clean_series.min()

                distribution['max'] = clean_series.max()

                distribution['mean'] = clean_series.mean()

                distribution['std'] = clean_series.std()

                distribution['median'] = clean_series.median()

                distribution['unique_ratio'] = len(clean_series.unique()) / len(clean_series)

        else:

            non_null = series.dropna()

            if len(non_null) > 0:

                distribution['unique_count'] = len(non_null.unique())

                distribution['unique_ratio'] = len(non_null.unique()) / len(non_null)

                distribution['most_common'] = non_null.value_counts().head(5).to_dict()



        return distribution



    def _detect_constraints(self, series: pd.Series) -> Dict:

        """Detect implicit constraints in the data"""

        constraints = {}



        # Missing value patterns

        constraints['null_ratio'] = series.isnull().mean()



        if pd.api.types.is_numeric_dtype(series):

            clean_series = pd.to_numeric(series, errors='coerce').dropna()

            if len(clean_series) > 0:

                constraints['always_positive'] = (clean_series >= 0).all()

                constraints['always_integer'] = (clean_series % 1 == 0).all()

                constraints['bounded_0_1'] = ((clean_series >= 0) & (clean_series <= 1)).all()

                constraints['bounded_0_100'] = ((clean_series >= 0) & (clean_series <= 100)).all()



        return constraints



    def _find_relationships(self, df: pd.DataFrame) -> Dict:

        """Find relationships and dependencies between columns"""

        relationships = {}



        # Find potential ID columns

        id_columns = []

        for col in df.columns:

            if (df[col].nunique() == len(df)) and pd.api.types.is_numeric_dtype(df[col]):

                id_columns.append(col)



        relationships['id_columns'] = id_columns



        # These are candidate relationships, not declared constraints. They are
        # used only to choose plausible cells for contextual corruption.

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        categorical_cols = df.select_dtypes(include=['object']).columns



        dependencies = {}



        # Numeric-Numeric relationships

        if len(numeric_cols) > 1:

            corr_matrix = df[numeric_cols].corr().abs()

            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):

                for j in range(i+1, len(corr_matrix.columns)):

                    if corr_matrix.iloc[i, j] > 0.7:  # High correlation threshold

                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

            dependencies['high_correlation'] = high_corr_pairs



        # Find columns that might be derived from others

        derived_relationships = []

        for col1 in df.columns:

            for col2 in df.columns:

                if col1 != col2:

                    # Check if col1 values appear to be derived from col2

                    if self._check_derivation(df[col1], df[col2]):

                        derived_relationships.append((col1, col2))



        dependencies['derived'] = derived_relationships

        relationships['dependencies'] = dependencies



        return relationships



    def _check_derivation(self, series1: pd.Series, series2: pd.Series) -> bool:

        """Check if series1 might be derived from series2"""

        # Simple heuristic: if series1 has patterns that match transformations of series2

        try:

            s1_str = series1.astype(str)

            s2_str = series2.astype(str)



            # Check if series1 contains substrings from series2

            contains_count = 0

            total_valid = 0



            for v1, v2 in zip(s1_str, s2_str):

                if pd.notna(v1) and pd.notna(v2) and len(str(v2)) > 2:

                    total_valid += 1

                    if str(v2).lower() in str(v1).lower():

                        contains_count += 1



            if total_valid > 0 and contains_count / total_valid > 0.8:

                return True



        except:

            pass



        return False



    def introduce_contextual_swap_errors(self, df: pd.DataFrame, error_rate: float, error_log=None) -> pd.DataFrame:

        """

        Introduce errors by swapping values between dependent attributes (cell-level)

        """

        # Preserve the caller's clean table because it is the ground-truth
        # reference used to evaluate the injected errors.
        df_copy = df.copy()

        if error_log is None:

            error_log = dict()

        analysis = self.analyze_data_structure(df_copy)

        n_rows, n_cols = df.shape

        total_cells = n_rows * n_cols

        n_errors = int(total_cells * error_rate)

        if n_errors > 0:

            # Generate all possible (row, col) cell indices

            all_cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]

            chosen_cells = random.sample(all_cells, min(n_errors, total_cells))

            relationships = analysis['relationships']['dependencies']

            for row_idx, col_idx in chosen_cells:

                col = df.columns[col_idx]

                clean_val = df.loc[row_idx, col]

                error_type = np.random.choice(['swap_dependent', 'cross_contamination', 'partial_swap'])

                desc = None

                if error_type == 'swap_dependent' and len(relationships.get('high_correlation', [])) > 0:

                    col1, col2 = random.choice(relationships['high_correlation'])

                    if col == col1 or col == col2:

                        other_idx = random.choice([i for i in range(n_rows) if i != row_idx])

                        new_val = df_copy.loc[other_idx, col]

                        df_copy.loc[row_idx, col] = new_val

                        desc = f"contextual_swaps: Value swapped with row {other_idx} in highly correlated column ({col1}, {col2})"

                elif error_type == 'cross_contamination':

                    cols = list(df.columns)

                    if len(cols) >= 2:

                        source_col = random.choice(cols)

                        target_col = random.choice([c for c in cols if c != source_col])

                        if col == target_col:

                            other_idx = random.choice([i for i in range(n_rows) if i != row_idx])

                            new_val = df_copy.loc[other_idx, source_col]

                            df_copy.loc[row_idx, target_col] = new_val

                            desc = f"contextual_swaps: Value from {source_col} in row {other_idx} copied to {target_col}"

                elif error_type == 'partial_swap':

                    text_cols = [c for c in df.columns if analysis['column_types'][c] in ['text', 'mixed_alphanumeric']]

                    if len(text_cols) >= 2 and col in text_cols:

                        col1 = col

                        col2 = random.choice([c for c in text_cols if c != col1])

                        val1 = str(df_copy.loc[row_idx, col1])

                        val2 = str(df_copy.loc[row_idx, col2])

                        if len(val1) > 3 and len(val2) > 3:

                            part1 = val1[:len(val1)//2]

                            part2 = val2[len(val2)//2:]

                            new_val = part1 + part2

                            df_copy.loc[row_idx, col1] = new_val

                            desc = f"contextual_swaps: Partial value from {col2} mixed into {col1}"

                if desc:

                    error_log[(row_idx, col)] = (clean_val, df_copy.loc[row_idx, col], desc)

        return df_copy

        return df_copy



    def introduce_pattern_drift_errors(self, df: pd.DataFrame, error_rate: float, error_log=None) -> pd.DataFrame:

        """

        Introduce errors by slightly violating discovered patterns (cell-level)

        """

        # Prefer small edits so that pattern violations remain realistic rather
        # than becoming trivially detectable corruptions.
        df_copy = df.copy()

        if error_log is None:

            error_log = dict()

        analysis = self.analyze_data_structure(df_copy)

        n_rows, n_cols = df.shape

        total_cells = n_rows * n_cols

        n_errors = int(total_cells * error_rate)

        if n_errors > 0:

            all_cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]

            chosen_cells = random.sample(all_cells, min(n_errors, total_cells))

            for row_idx, col_idx in chosen_cells:

                col = df.columns[col_idx]

                clean_val = df.loc[row_idx, col]

                col_type = analysis['column_types'][col]

                current_value = str(df_copy.loc[row_idx, col])

                desc = None

                if col_type == 'code' and len(current_value) > 1:

                    chars = list(current_value)

                    pos = np.random.randint(0, len(chars))

                    if chars[pos].isalpha():

                        chars[pos] = np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

                        desc = "pattern_drift: Random alphabetic character replaced"

                    elif chars[pos].isdigit():

                        chars[pos] = str(np.random.randint(0, 9))

                        desc = "pattern_drift: Random digit replaced"

                    df_copy.loc[row_idx, col] = ''.join(chars)

                elif col_type == 'mixed_alphanumeric':

                    self._apply_character_substitution(df_copy, row_idx, col, current_value)

                    desc = "pattern_drift: Character visually or keyboard substituted"

                elif col_type == 'integer_id':

                    try:

                        num_val = int(current_value)

                        error_type = np.random.choice(['off_by_one', 'digit_error', 'missing_digit'])

                        if error_type == 'off_by_one':

                            df_copy.loc[row_idx, col] = num_val + np.random.choice([-1, 1])

                            desc = "pattern_drift: Off-by-one error"

                        elif error_type == 'digit_error' and len(str(num_val)) > 1:

                            str_val = str(num_val)

                            pos = np.random.randint(0, len(str_val))

                            new_digit = str(np.random.randint(0, 9))

                            new_str = str_val[:pos] + new_digit + str_val[pos+1:]

                            df_copy.loc[row_idx, col] = int(new_str)

                            desc = "pattern_drift: Random digit replaced in integer"

                        elif error_type == 'missing_digit' and len(str(num_val)) > 2:

                            str_val = str(num_val)

                            pos = np.random.randint(0, len(str_val))

                            new_str = str_val[:pos] + str_val[pos+1:]

                            df_copy.loc[row_idx, col] = int(new_str) if new_str else num_val

                            desc = "pattern_drift: Digit removed from integer"

                    except:

                        pass

                if desc:

                    error_log[(row_idx, col)] = (clean_val, df_copy.loc[row_idx, col], desc)

        return df_copy



    def _apply_character_substitution(self, df_copy: pd.DataFrame, idx: int, col: str, value: str):

        """Apply character substitution based on similarity patterns"""

        for char, substitutes in self.char_substitutions.items():

            if char in value:

                substitute = np.random.choice(substitutes)

                # Replace only first occurrence

                new_value = value.replace(char, substitute, 1)

                df_copy.loc[idx, col] = new_value

                break



    def introduce_distribution_outliers(self, df: pd.DataFrame, error_rate: float, error_log=None) -> pd.DataFrame:

        """

        Introduce outliers based on discovered distributions (cell-level)

        """

        df_copy = df.copy()

        if error_log is None:

            error_log = dict()

        analysis = self.analyze_data_structure(df_copy)

        n_rows, n_cols = df.shape

        total_cells = n_rows * n_cols

        n_errors = int(total_cells * error_rate)

        if n_errors > 0:

            all_cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]

            chosen_cells = random.sample(all_cells, min(n_errors, total_cells))

            for row_idx, col_idx in chosen_cells:

                col = df.columns[col_idx]

                clean_val = df.loc[row_idx, col]

                if col in analysis['distributions'] and 'mean' in analysis['distributions'][col]:

                    dist = analysis['distributions'][col]

                    mean = dist.get('mean', 0)

                    std = dist.get('std', 1)

                    if std > 0:

                        outlier_direction = np.random.choice([-1, 1])

                        outlier_magnitude = np.random.uniform(3, 5)

                        outlier_value = mean + (outlier_direction * outlier_magnitude * std)

                        constraints = analysis['constraints'][col]

                        if constraints.get('always_positive', False):

                            outlier_value = max(0, outlier_value)

                        if constraints.get('bounded_0_1', False):

                            outlier_value = max(0, min(1, outlier_value))

                        df_copy.loc[row_idx, col] = outlier_value

                        desc = "distribution_outliers: Outlier value injected based on distribution"

                        error_log[(row_idx, col)] = (clean_val, outlier_value, desc)

        return df_copy



    def introduce_encoding_corruptions(self, df: pd.DataFrame, error_rate: float, error_log=None) -> pd.DataFrame:

        """

        Introduce encoding, formatting, and keyboard-based corruptions (cell-level)

        """

        df_copy = df.copy()

        if error_log is None:

            error_log = dict()

        n_rows, n_cols = df.shape

        total_cells = n_rows * n_cols

        n_errors = int(total_cells * error_rate)

        if n_errors > 0:

            all_cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]

            chosen_cells = random.sample(all_cells, min(n_errors, total_cells))

            # Retain these Unicode variants for optional experiments. The
            # active branch below currently applies keyboard-neighbour typos.
            encoding_errors = {

                # Unicode normalization errors

                'a': ['ª', 'à', 'á', 'â', 'ã', 'ä', 'å'],

                'e': ['è', 'é', 'ê', 'ë'],

                'i': ['ì', 'í', 'î', 'ï'],

                'o': ['ò', 'ó', 'ô', 'õ', 'ö'],

                'u': ['ù', 'ú', 'û', 'ü'],

                'n': ['ñ'],

                'c': ['ç'],

                # Punctuation errors

                '-': ['–', '—', '−'],

                "'": ['‘', '’', '`'],

                '"': ['“', '”', '``'],

                '...': ['…'],

            }

            for row_idx, col_idx in chosen_cells:

                col = df.columns[col_idx]

                clean_val = df.loc[row_idx, col]

                if df[col].dtype == 'object':

                    value = str(df_copy.loc[row_idx, col])

                    desc = None

                    # if random.random() < 0.5:

                    #     for char, variants in encoding_errors.items():

                    #         if char in value:

                    #             variant = np.random.choice(variants)

                    #             df_copy.loc[row_idx, col] = value.replace(char, variant, 1)

                    #             desc = "encoding_corruptions: Unicode or punctuation variant substituted"

                    #             break

                    # else:

                    corrupted = butterfinger(value, prob=0.6)

                    if corrupted != value:

                        df_copy.loc[row_idx, col] = corrupted

                        desc = "encoding_corruptions: Keyboard-based typo introduced"

                    if desc:

                        error_log[(row_idx, col)] = (clean_val, df_copy.loc[row_idx, col], desc)

        return df_copy



    def introduce_measurement_noise(self, df: pd.DataFrame, error_rate: float, error_log=None) -> pd.DataFrame:

        """

        Introduce subtle measurement noise and rounding errors (cell-level)

        """

        df_copy = df.copy()

        if error_log is None:

            error_log = dict()

        analysis = self.analyze_data_structure(df_copy)

        n_rows, n_cols = df.shape

        total_cells = n_rows * n_cols

        n_errors = int(total_cells * error_rate)

        if n_errors > 0:

            all_cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]

            chosen_cells = random.sample(all_cells, min(n_errors, total_cells))

            for row_idx, col_idx in chosen_cells:

                col = df.columns[col_idx]

                clean_val = df.loc[row_idx, col]

                if pd.api.types.is_numeric_dtype(df[col]):

                    current_value = df_copy.loc[row_idx, col]

                    if pd.notna(current_value) and current_value != 0:

                        # A narrow perturbation approximates measurement or
                        # rounding noise without creating an obvious outlier.
                        noise_factor = np.random.uniform(-0.02, 0.02)

                        noise = current_value * noise_factor

                        new_value = current_value + noise

                        constraints = analysis['constraints'][col]

                        if constraints.get('always_positive', False):

                            new_value = max(0, new_value)

                        if constraints.get('always_integer', False):

                            new_value = round(new_value)

                        df_copy.loc[row_idx, col] = new_value

                        desc = "measurement_noise: Small random noise added"

                        error_log[(row_idx, col)] = (clean_val, new_value, desc)

        return df_copy



    def inject_errors(self, input_file: str, output_file: str, error_rates: Dict[str, float], error_report_file: str = None) -> None:

        """

        Main function to inject domain-agnostic errors



        Args:

            input_file (str): Path to input CSV file

            output_file (str): Path to output CSV file with errors

            error_rates (dict): Dictionary with error types and their rates

                - 'contextual_swaps': float (0.0 to 1.0)

                - 'pattern_drift': float (0.0 to 1.0)

                - 'distribution_outliers': float (0.0 to 1.0)

                - 'encoding_corruptions': float (0.0 to 1.0)

                - 'measurement_noise': float (0.0 to 1.0)

        """

        # Read and analyze the data

        print(f"Reading and analyzing CSV file: {input_file}")

        df = pd.read_csv(input_file)

        original_shape = df.shape



        print(f"Original dataset shape: {original_shape}")



        # Analyze data structure

        print(" Analyzing data structure and discovering patterns...")

        analysis = self.analyze_data_structure(df)



        print(f" Detected {len(analysis['column_types'])} columns with various types")

        print(f" Found {len(analysis['relationships']['dependencies'].get('high_correlation', []))} high-correlation pairs")

        print(f" Identified {len(analysis['relationships']['id_columns'])} potential ID columns")



        # Keying by (row, column) means that if two injectors select the same
        # cell, the report retains its final corrupted value.

        error_log = dict()

        if error_rates.get('contextual_swaps', 0) > 0:

            print(f" Introducing contextual swap errors (rate: {error_rates['contextual_swaps']:.2%})")

            df = self.introduce_contextual_swap_errors(df, error_rates['contextual_swaps'], error_log)

        if error_rates.get('pattern_drift', 0) > 0:

            print(f" Introducing pattern drift errors (rate: {error_rates['pattern_drift']:.2%})")

            df = self.introduce_pattern_drift_errors(df, error_rates['pattern_drift'], error_log)

        if error_rates.get('distribution_outliers', 0) > 0:

            print(f" Introducing distribution-based outliers (rate: {error_rates['distribution_outliers']:.2%})")

            df = self.introduce_distribution_outliers(df, error_rates['distribution_outliers'], error_log)

        if error_rates.get('encoding_corruptions', 0) > 0:

            print(f" Introducing encoding corruptions (rate: {error_rates['encoding_corruptions']:.2%})")

            df = self.introduce_encoding_corruptions(df, error_rates['encoding_corruptions'], error_log)

        if error_rates.get('measurement_noise', 0) > 0:

            print(f" Introducing measurement noise (rate: {error_rates['measurement_noise']:.2%})")

            df = self.introduce_measurement_noise(df, error_rates['measurement_noise'], error_log)

        # Save error report if requested

        if error_report_file is not None:

            import csv

            with open(error_report_file, 'w', newline='', encoding='utf-8') as f:

                writer = csv.writer(f)

                writer.writerow(['row', 'column', 'clean_value', 'error_value', 'error_type'])

                for (row_idx, col), (clean_val, error_val, desc) in error_log.items():

                    writer.writerow([row_idx, col, clean_val, error_val, desc])



        # Cell corruption must not add or remove tuples or attributes.

        final_shape = df.shape

        print(f"Final dataset shape: {final_shape}")



        if original_shape != final_shape:

            print("WARNING: Dataset shape changed! This should not happen.")

        else:

            print("✓ Dataset shape preserved")



        # Save to output file

        df.to_csv(output_file, index=False)

        print(f"Universal corrupted dataset saved to: {output_file}")



        # Print summary

        self._print_analysis_summary(analysis, error_rates, original_shape)



    def _print_analysis_summary(self, analysis: Dict, error_rates: Dict, shape: Tuple):

        """Print comprehensive analysis and error injection summary"""

        print("\n" + "="*60)

        print(" UNIVERSAL ERROR INJECTION SUMMARY")

        print("="*60)



        print(f" Dataset: {shape[0]:,} rows × {shape[1]} columns")



        print("\n Discovered Column Types:")

        for col_type, count in pd.Series(list(analysis['column_types'].values())).value_counts().items():

            print(f"   • {col_type}: {count} columns")



        print("\n Discovered Relationships:")

        deps = analysis['relationships']['dependencies']

        print(f"   • High correlations: {len(deps.get('high_correlation', []))} pairs")

        print(f"   • Derived relationships: {len(deps.get('derived', []))} pairs")

        print(f"   • ID columns: {len(analysis['relationships']['id_columns'])} columns")



        print("\n Error Injection Results:")

        total_rows = shape[0]

        total_cols = shape[1]

        total_cells = total_rows * total_cols

        for error_type, rate in error_rates.items():

            if rate > 0:

                affected_cells = int(total_cells * rate)

                print(f"   • {error_type}: ~{affected_cells:,} cells affected ({rate:.2%})")





# Example usage function

def example_usage():

    """

    Example of how to use the Universal Error Injector

    """

    # Initialize the universal error injector

    injector = UniversalErrorInjector(seed=42)



    # Define universal error rates

    error_rates = {

        'contextual_swaps': 0.05,          # 5% - Swap dependent values

        'pattern_drift': 0.05,            # 5% - Violate discovered patterns

        'distribution_outliers': 0.04,    # 4% - Statistical outliers

        'encoding_corruptions': 0.04,      # 4% - Character encoding issues

        'measurement_noise': 0.03          # 3% - Subtle measurement errors

    }



    # Inject errors (works with ANY dataset!)

    input_file = "flights-trial/clean.csv"  

    output_file = "flights-trial/dirty.csv"

    error_report_file = "flights-trial/error_report.csv"



    try:

        injector.inject_errors(input_file, output_file, error_rates, error_report_file=error_report_file)

        print("\n Universal error injection completed successfully!")

        print(" This approach works on ANY dataset without domain knowledge!")



    except FileNotFoundError:

        print(f"Error: Input file '{input_file}' not found.")

        print("Please ensure the file exists in the current directory.")

    except Exception as e:

        print(f"An error occurred: {str(e)}")





if __name__ == "__main__":

    example_usage()
