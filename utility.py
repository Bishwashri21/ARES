import os
import pandas as pd
import logging
import time
import traceback
from openai import OpenAI
from config import DATA_DIR
from llm_logging import log_llm_event


class Dataset:
    """
    Represents one dataset and keeps both its dirty and clean versions.

    The dirty dataset is used by the error-detection pipeline.
    The clean dataset is used as ground truth to identify which cells
    are actually erroneous.
    """

    name: str
    directory: str
    dirty_df: pd.DataFrame
    clean_df: pd.DataFrame

    def __init__(self, name, data_dir=DATA_DIR):
        # Store the dataset name and the directory containing the data files.
        self.name = name
        self.directory = data_dir

        # ---------------------------------------------------------------
        # Locate the clean and dirty files.
        #
        # First, use the ZeroED naming convention:
        #
        #   flights_clean.csv
        #   flights_error-01.csv
        #
        # for a dataset named "flights".
        # ---------------------------------------------------------------
        clean_path = os.path.join(
            self.directory,
            f"{self.name}_clean.csv"
        )

        dirty_path = os.path.join(
            self.directory,
            f"{self.name}_error-01.csv"
        )

        # ---------------------------------------------------------------
        # If the above files do not exist, fall back to the SAGED-style
        # directory structure:
        #
        #   data/
        #       flights/
        #           clean.csv
        #           dirty.csv
        # ---------------------------------------------------------------
        if not os.path.exists(clean_path):
            clean_path = os.path.join(
                self.directory,
                self.name,
                "clean.csv"
            )

        if not os.path.exists(dirty_path):
            dirty_path = os.path.join(
                self.directory,
                self.name,
                "dirty.csv"
            )

        # Stop the program if the dirty file cannot be found.
        if not os.path.exists(dirty_path):
            raise FileNotFoundError(
                f"Couldn't find dirty file for dataset "
                f"{self.name} at {dirty_path}"
            )

        # Stop the program if the clean file cannot be found.
        if not os.path.exists(clean_path):
            raise FileNotFoundError(
                f"Couldn't find clean file for dataset "
                f"{self.name} at {clean_path}"
            )

        # ---------------------------------------------------------------
        # Load both datasets as strings.
        #
        # dtype=str is intentional because dirty data can contain values
        # such as "25", "25.0", "unknown", "?", etc.
        # Reading everything as strings avoids unwanted automatic type
        # conversion by pandas.
        # ---------------------------------------------------------------
        self.dirty_df = pd.read_csv(
            dirty_path,
            low_memory=False,
            dtype=str
        )

        self.clean_df = pd.read_csv(
            clean_path,
            low_memory=False,
            dtype=str
        )

        # Make the dirty dataframe use exactly the same column names
        # as the clean dataframe.
        self.dirty_df.columns = self.clean_df.columns

        # ---------------------------------------------------------------
        # Ensure the dirty and clean datasets have compatible sizes.
        #
        # The error-detection code compares dirty row i with clean row i,
        # so both datasets need to have the same number of rows.
        # ---------------------------------------------------------------
        if self.dirty_df.shape != self.clean_df.shape:

            # Use the smaller number of rows if their lengths differ.
            min_len = min(
                len(self.dirty_df),
                len(self.clean_df)
            )

            # Keep only the first min_len rows of each dataset.
            self.dirty_df = self.dirty_df.iloc[:min_len]
            self.clean_df = self.clean_df.iloc[:min_len]

        # ---------------------------------------------------------------
        # Reset both indices so that:
        #
        # dirty row 0 <-> clean row 0
        # dirty row 1 <-> clean row 1
        # ...
        #
        # drop=True means the old index is discarded instead of becoming
        # a new dataframe column.
        # ---------------------------------------------------------------
        self.dirty_df = self.dirty_df.reset_index(drop=True)
        self.clean_df = self.clean_df.reset_index(drop=True)

    def get_actual_errors(self):
        """
        Compare the dirty and clean datasets cell by cell.

        Returns:
            DataFrame where:
                1 = the dirty value is different from the clean value
                0 = the dirty value is the same as the clean value
        """

        # Find columns that are present in both dataframes.
        common_cols = self.dirty_df.columns.intersection(
            self.clean_df.columns
        )

        # Keep only those common columns.
        d_df = self.dirty_df[common_cols]
        c_df = self.clean_df[common_cols]

        # Compare the two tables cell by cell.
        #
        # True  -> dirty value and clean value are different
        # False -> they are the same
        #
        # astype(int) converts:
        #     True  -> 1
        #     False -> 0
        errors = (d_df != c_df).astype(int)

        return errors

    def __str__(self):
        # Defines what should be displayed when the Dataset object
        # is printed.
        #
        # Example:
        #     flights (5000, 12)
        return f"{self.name} {self.dirty_df.shape}"


class Timer:
    """
    Simple context-manager used to measure how long an operation takes.

    Example:
        with Timer("Feature generation"):
            generate_features()
    """

    def __init__(self, name, logger=None):
        # Store the name of the operation being timed.
        self.name = name

        # A logger is optional.
        # If no logger is provided, messages are printed to the terminal.
        self.logger = logger

    def __enter__(self):
        # Record the time when the operation starts.
        self.start = time.time()

        # Report that the operation has started.
        if self.logger:
            self.logger.info(f"{self.name}......")
        else:
            print(f"{self.name}......")

        # Return this Timer object.
        # This allows:
        #
        #     with Timer("Test") as timer:
        #
        # where timer refers to this object.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Record the time when the operation finishes.
        self.end = time.time()

        # Calculate how many seconds the operation took.
        self.duration = self.end - self.start

        # ---------------------------------------------------------------
        # Check whether an exception occurred inside the "with" block.
        #
        # exc_type is None  -> no exception
        # exc_type is not None -> an exception occurred
        # ---------------------------------------------------------------
        if exc_type is not None:

            # Log the error if a logger was provided.
            if self.logger:
                self.logger.error(
                    f"Error in {self.name}: {exc_val}"
                )

                # Write the complete traceback to the log.
                self.logger.error(
                    f"Traceback:\n{traceback.format_exc()}"
                )

            # Otherwise, print the error.
            else:
                print(
                    f"Error in {self.name}: {exc_val}"
                )

            # False means that the exception should NOT be suppressed.
            # The original exception will continue propagating.
            return False

        # ---------------------------------------------------------------
        # If no exception occurred, report successful completion and
        # the amount of time taken.
        # ---------------------------------------------------------------
        if self.logger:
            self.logger.info(
                f"Finish {self.name}, Using {self.duration}s\n"
            )
        else:
            print(
                f"Finish {self.name}, Using {self.duration}s\n"
            )

        # Return the elapsed time.
        return self.duration


def get_logger(name, log_file):
    """
    Create and configure a logger that writes to both:
        1. the terminal
        2. a log file
    """

    # Get the logger identified by the given name.
    logger = logging.getLogger(name)

    # If handlers were already attached to this logger, remove them.
    # This prevents duplicate messages when the logger is configured
    # more than once.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Record INFO-level messages and anything more severe.
    logger.setLevel(logging.INFO)

    # ---------------------------------------------------------------
    # Create a handler for terminal/console output.
    # ---------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # ---------------------------------------------------------------
    # Create a handler for writing messages to a file.
    # ---------------------------------------------------------------
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # ---------------------------------------------------------------
    # Define the format of every log message.
    #
    # Example:
    # 2026-09-23 12:00:00 - ARES - INFO - Starting pipeline
    # ---------------------------------------------------------------
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Apply the same format to both handlers.
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach both handlers to the logger.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Return the fully configured Logger object.
    return logger


def get_ans_from_llm(prompt, api_use=False):
    """
    Send a prompt to an LLM and return the generated response.

    There are two possible modes:

    1. api_use=False:
       Use the locally hosted Mistral model.

    2. api_use=True:
       Use the remote OpenAI-compatible API.
    """

    # ===============================================================
    # MODE 1: LOCAL LLM
    # ===============================================================
    if not api_use:

        # The local server does not require a real OpenAI API key.
        openai_api_key = "EMPTY"

        # Address of the locally hosted OpenAI-compatible server.
        openai_api_base = "http://localhost:8000/v1"

        # Number of times to retry if the local request fails.
        max_retries = 5

        # Base sleep value.
        # NOTE: It is currently defined but not used in this branch.
        base_sleep = 1

        # Path/name of the locally hosted Mistral model.
        model_name = (
            "/u/student/2024/cs24mresch11013/"
            "ares-main/models/Mistral-7B-Instruct-v0.3"
        )

        # Try the request up to max_retries times.
        for attempt in range(max_retries):

            try:
                # Create an OpenAI-compatible client that points
                # to the local inference server.
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                    max_retries=100,
                )

                # ---------------------------------------------------
                # Send the actual prompt to the local LLM.
                # ---------------------------------------------------
                chat_response = client.chat.completions.create(
                    model=model_name,

                    messages=[
                        # System message defines the general role
                        # of the model.
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },

                        # User message contains the actual prompt
                        # generated by the ARES/ZeroED pipeline.
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],

                    # Temperature 0 makes generation as deterministic
                    # as possible.
                    temperature=0,

                    # Limit the generated answer to 1024 tokens.
                    max_tokens=1024,

                    # Additional generation parameter supported by
                    # the local inference server.
                    extra_body={
                        "repetition_penalty": 1.05,
                    },
                )

                # ---------------------------------------------------
                # Check whether the server returned at least one
                # response choice.
                # ---------------------------------------------------
                if (
                    chat_response
                    and chat_response.choices
                    and len(chat_response.choices) > 0
                ):

                    # Extract the actual text generated by the model.
                    response_text = (
                        chat_response
                        .choices[0]
                        .message
                        .content
                    )

                    # Try to get token-usage information from the
                    # response, if the server provides it.
                    usage = getattr(
                        chat_response,
                        "usage",
                        None
                    )

                    # Record this LLM call for token/usage tracking.
                    log_llm_event(
                        model_name,
                        prompt,
                        response_text,
                        usage=usage
                    )

                    # Return the generated response to the caller.
                    return response_text

                else:
                    # The request succeeded, but the server returned
                    # no usable answer.
                    print("Warning: Empty response from LLM")

                    # Record the failed/empty response in the LLM log.
                    log_llm_event(
                        model_name,
                        prompt,
                        "",
                        usage=None
                    )

                    return ""

            except Exception as e:

                # Print the error so it is visible in the terminal.
                print(e)

                # Record the failed LLM call.
                log_llm_event(
                    model_name,
                    prompt,
                    "",
                    usage=None
                )

                # If this was the final retry, stop trying.
                if attempt == max_retries - 1:
                    print(
                        f"Failed after {max_retries} attempts"
                    )
                    return ""

    # ===============================================================
    # MODE 2: REMOTE API
    # ===============================================================
    elif api_use:

        # Name of the model requested from the remote API.
        model_type = "qwen2.5-7b-instruct"

        # System instruction defining the role of the remote model.
        role_descr = (
            "You are a world-class data engineer, "
            "proficient in cleaning dirty data."
        )

        # The code is structured to support multiple API keys.
        #
        # NOTE: The current value is empty, so this branch will not
        # have a usable API key unless it is filled in.
        api_key_list = [
            ''
        ]

        # -----------------------------------------------------------
        # Retry-related settings.
        # -----------------------------------------------------------
        base_sleep = 0.2
        max_retries = 200

        # Number of retry cycles completed so far.
        try_cnt = 0

        # Index of the API key currently being used.
        key_idx = 0

        # Keep trying until the request succeeds or the retry limit
        # is reached.
        while 1:

            try:
                # Create the OpenAI client for the remote API.
                client = OpenAI(
                    api_key=api_key_list[key_idx],
                    base_url="https://api.openai.com/v1"
                )

                # ---------------------------------------------------
                # Send the prompt to the remote model.
                # ---------------------------------------------------
                completion_res = client.chat.completions.create(
                    model=model_type,

                    # Use deterministic generation.
                    temperature=0,

                    messages=[
                        # System instruction.
                        {
                            "role": "system",
                            "content": role_descr
                        },

                        # Actual user prompt.
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                # Extract the generated response text.
                response = (
                    completion_res
                    .choices[0]
                    .message
                    .content
                )

                # Read token usage if it is available.
                usage = getattr(
                    completion_res,
                    "usage",
                    None
                )

                # Record the successful LLM call.
                log_llm_event(
                    model_type,
                    prompt,
                    response,
                    usage=usage
                )

                # Reset retry counters after a successful request.
                try_cnt = 0
                key_idx = 0

                # Return the generated response.
                return response

            except Exception as e:

                # Print the API error.
                print(e)

                # Record the failed request.
                log_llm_event(
                    model_type,
                    prompt,
                    "",
                    usage=None
                )

                # Move to the next API key.
                key_idx += 1

                # If all available API keys have been tried,
                # wait before starting another retry cycle.
                if key_idx >= len(api_key_list):

                    # Calculate how long to wait before retrying.
                    sleep_time = (
                        base_sleep * (2 + try_cnt)
                    )

                    # Pause the program.
                    time.sleep(sleep_time)

                    # Display the waiting time.
                    print(
                        "Sleeping for {} seconds".format(
                            sleep_time
                        )
                    )

                    # Print the original error again.
                    print(e)

                    # Increase the retry-cycle counter.
                    try_cnt += 1

                    # Start again with the first API key.
                    key_idx = 0

                    # Stop once the maximum number of retry cycles
                    # has been reached.
                    if try_cnt >= max_retries:
                        print(
                            "Maximum {} retries reached "
                            "for OpenAI API requests.".format(
                                max_retries
                            )
                        )
                        break

    # If neither LLM mode succeeds, return an empty string.
    #
    # This prevents callers from receiving None and helps avoid
    # TypeError when the response is later processed as text.
    return ""