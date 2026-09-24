import pandas as pd
import numpy as np
import re
import ast
import random
import json
from collections import Counter
from utility import get_ans_from_llm

class LLMAnalyzer:
    def __init__(self, dirty_df):
        self.dirty_df = dirty_df

    def get_column_examples(self, attr_name, n_samples=10):
        if attr_name not in self.dirty_df.columns:
            return ""
        # Get non-null samples
        non_null_series = self.dirty_df[attr_name].dropna()
        if non_null_series.empty:
            return "[]"
        
        n = min(n_samples, len(non_null_series))
        samples = non_null_series.sample(n=n).tolist()
        return str(samples)

    def extract_code(self, text):
        """Extracts python code blocks from text."""
        code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    def generate_distribution_analysis(self, attr_name):
        examples = self.get_column_examples(attr_name)
        prompt = f"""
Based on the column '{attr_name}' with examples: {examples}

Please generate Python functions to analyze the data distribution from various perspectives, so that we can verify whether an error is reasonable or not. 
Each function should:
1. Take parameters (dirty_csv: dataframe, attr_name: str), regard all values in dirty_csv are **strings**
2. Return a string containing the **detailed** analysis results
3. Do not enumerate/count all values, showing representative ones
4. **Also import necessary libraries** inside the function or at the top.

Example function code snippet:
```python 
def distr_analysis_perspective(dirty_csv, attr_name):
    import pandas as pd
    # Your logic here
    return 'Detailed description of the analysis results'
```

Provide your functions below:
        """
        response = get_ans_from_llm(prompt)
        code_blocks = self.extract_code(response)
        
        analysis_results = []
        
        for code in code_blocks:
            try:
                # Create a safe local scope
                local_scope = {}
                # Execute the code definition
                exec(code, globals(), local_scope)
                
                # Find the function name - relax the check to find any new callable
                func_name = None
                for key, val in local_scope.items():
                    # Check if it's a function and not an import or builtin
                    if callable(val) and not key.startswith("__"):
                        # Prefer functions with 'distr' or 'analy' in name, but accept others if only one exists
                        func_name = key
                        # If we find one that matches the pattern, break. Otherwise keep looking/overwriting
                        if "distr" in key or "analy" in key:
                            break
                
                if func_name:
                    # Execute the function
                    res = local_scope[func_name](self.dirty_df, attr_name)
                    analysis_results.append(str(res))
            except Exception as e:
                # print(f"Error executing distribution analysis code for {attr_name}: {e}")
                pass
                
        if not analysis_results:
            # Fallback: Just use the LLM response text if it contains analysis
            # Sometimes LLM explains instead of writing code
            if len(response) > 100 and "def " not in response:
                 return response, prompt, response
            return f"Could not generate detailed analysis. Examples: {examples}", prompt, response
            
        return "\n".join(analysis_results), prompt, response

    def generate_guidelines(self, attr_name, dist_analysis):
        examples = self.get_column_examples(attr_name)
        prompt = f"""
You are a top data guideline generation scientist in prompt data cleaning. Please generate a comprehensive guide for identifying and analyzing common errors in the '{attr_name}' attribute.

Here are the data distribution analysis results for the attribute '{attr_name}':
{dist_analysis}

Here are some examples for '{attr_name}':
{examples}

Please first explain the meaning of attribute '{attr_name}'.

Then, for each error type below, considering the data distribution analysis results, provide specific causes, examples, and detection methods for '{attr_name}':
1. Pattern violations - examples: <Desc>
   - causes: <Desc>
   - det methods: <Desc>
2. Missing values - examples: <Desc>
   - causes: <Desc>
   - det methods: <Desc>
3. Rule violations - examples: <Desc>
   - causes: <Desc>
   - det methods: <Desc>
4. Outliers - examples: <Desc>
   - causes: <Desc>
   - det methods: <Desc>
5. Typos - examples: <Desc>
   - causes: <Desc>
   - det methods: <Desc>

By systematically identifying these errors, you can ensure the attribute data in the table is more clean for further analysis.
        """
        response = get_ans_from_llm(prompt)
        return response, prompt

    def generate_criteria_functions(self, attr_name, guidelines):
        prompt = f"""
Based on the following error detection guidelines for attribute '{attr_name}':
{guidelines}

Please generate Python functions to detect these errors.
Each function should:
1. Take parameters (val, row) where val is the cell value and row is the pandas Series for the row.
2. Return True if the value is an error, False otherwise.
3. Handle potential exceptions (e.g., type conversion) gracefully and return True (error) or False (clean) as appropriate.
4. Be specific to the attribute '{attr_name}'.

Example:
```python
def check_pattern_violation(val, row):
    import re
    if pd.isna(val): return True
    if not re.match(r'^\d+$', str(val)):
        return True
    return False
```

Provide the functions in Python code blocks.
        """
        response = get_ans_from_llm(prompt)
        code_blocks = self.extract_code(response)
        return code_blocks, prompt, response

    def label_samples(self, samples, attr_name, guidelines, use_confidence=True):
        # samples: list of (index, row_dict)
        
        # Format samples for the prompt
        entries = []
        for idx, row_dict in samples:
            # Ensure dict is JSON serializable (handle NaNs, Timestamps, and other non-serializable types)
            safe_dict = {}
            for k, v in row_dict.items():
                try:
                    if pd.isna(v):
                        safe_dict[k] = None
                        continue
                except (TypeError, ValueError):
                    pass
                safe_dict[k] = str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
            entries.append(f"Index {idx}: {json.dumps(safe_dict)}")
        samples_str = "\n".join(entries)
        
        prompt = f"""
As a data quality expert, please analyze the '{attr_name}' attribute values for potential errors based on the provided guidelines.
Guidelines:
{guidelines}

Provide your analysis on `{attr_name}` values in JSON format as follows:
```json
{{
  "entries": [
    {{
      "index": <original_index>,
      "value": "<value_of_{attr_name}>",
      "error_analysis": "<Brief explanation>",
      "has_error": true/false,
      "confidence": "High" or "Medium" or "Low"
    }},
    ...
  ]
}}
```

Here are the samples to analyze:
{samples_str}

Only return the JSON.
        """
        response = get_ans_from_llm(prompt)
        
        final_labels = {}
        try:
            json_str = None
            # 1. Check for markdown code blocks
            code_blocks = self.extract_code(response)
            if code_blocks:
                for block in code_blocks:
                    try:
                        # Try to parse
                        parsed = json.loads(block)
                        if "entries" in parsed:
                            json_str = block
                            break
                    except: continue
            
            # 2. If no valid JSON block, try regex for the outer object
            if not json_str:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
            
            # 3. If still nothing, try the whole response
            if not json_str:
                json_str = response

            if json_str:
                data = None
                e1 = e2 = e3 = e4 = None
                try:
                    data = json.loads(json_str)
                except Exception as exc1:
                    e1 = exc1
                    try:
                        # Try decoding common escaped sequences
                        fixed = json_str.encode('utf-8').decode('unicode_escape')
                        data = json.loads(fixed)
                    except Exception as exc2:
                        e2 = exc2
                        try:
                            # Try swapping single quotes to double quotes (common LLM output)
                            alt = re.sub(r"(?<!\\)'", '"', json_str)
                            data = json.loads(alt)
                        except Exception as exc3:
                            e3 = exc3
                            try:
                                # Last resort: eval-ish parsing via ast (may accept Python literals)
                                data = ast.literal_eval(json_str)
                            except Exception as exc4:
                                e4 = exc4

                if data is None:
                    print(f"Warning: Parsed JSON but found no valid labels for {attr_name} (errors: {e1}, {e2}, {e3}, {e4})")
                else:
                    if "entries" in data:
                        low_confidence_labels = {}
                        for entry in data["entries"]:
                            idx = entry.get("index")
                            has_error = entry.get("has_error")
                            confidence = entry.get("confidence", "High") # Default to High if missing

                            if idx is None or has_error is None:
                                continue

                            label = 1 if has_error else 0
                            if use_confidence and str(confidence).lower() == "low":
                                # Keep separately; used as fallback if no higher-confidence labels exist
                                low_confidence_labels[int(idx)] = label
                            else:
                                # When use_confidence=False, all labels are accepted regardless of confidence
                                final_labels[int(idx)] = label

                        # If the LLM marked every sample as Low confidence, accept them anyway
                        # so that label propagation is not left empty.
                        if not final_labels and low_confidence_labels:
                            final_labels.update(low_confidence_labels)
                    else:
                        # Fallback for flat dict if LLM returned a mapping index->label
                        for k, v in data.items():
                            try:
                                if str(k).isdigit():
                                    final_labels[int(k)] = 1 if str(v).lower() in ["true", "error", "1", "yes"] else 0
                            except Exception:
                                continue

        except Exception as e:
            print(f"Error parsing labeling response for {attr_name}: {e}")
            
        return final_labels, prompt, response

    def generate_errors(self, attr_name, num_errors=5):
        # Get some clean examples (assuming majority are clean or using historical data if available)
        # For now, we just sample from the current dataset as "potential" clean/dirty mix
        examples = self.get_column_examples(attr_name, n_samples=10)
        
        prompt = f"""
You are a data quality analyst. Your task is to generate realistic errors for the attribute '{attr_name}'.

Here are some existing values from the dataset:
{examples}

Please generate {num_errors} realistic error examples for '{attr_name}' that could occur in real-world scenarios.
Cover different error types:
1. Pattern Violations
2. Missing Values (Explicit/Implicit)
3. Constraint Violations
4. Out-of-domain values
5. Typos
6. Common Knowledge Violations

Output format:
[
  ["{attr_name}", "error_value_1", "Reason: Error type - specific reason"],
  ["{attr_name}", "error_value_2", "Reason: Error type - specific reason"],
  ...
]
Only return the JSON list.
        """
        response = get_ans_from_llm(prompt)
        return response, prompt
