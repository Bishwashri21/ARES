import os
import json


# Stores the final human-readable LLM log file path for the current run.
LLM_LOG_PATH = None

# Stores the temporary JSONL file where we record information about
# each individual LLM call.
LLM_EVENTS_PATH = None


def set_llm_log_path(log_path):
    """
    Configure the LLM logging paths for the current pipeline run.

    The final summary will be written to `log_path`, while individual
    LLM-call records will be temporarily stored in `llm_events.jsonl`
    in the same directory.
    """
    global LLM_LOG_PATH, LLM_EVENTS_PATH

    # Save the path of the final summary file.
    LLM_LOG_PATH = log_path

    # Extract the directory containing the final log file.
    base_dir = os.path.dirname(log_path) if log_path else None

    # Create the path for the temporary per-call event file.
    # Example:
    #   log_path      = results/run1/llm_logging.txt
    #   events path   = results/run1/llm_events.jsonl
    LLM_EVENTS_PATH = (
        os.path.join(base_dir, "llm_events.jsonl")
        if base_dir
        else None
    )

    # Create the event file and its parent directory if necessary.
    if LLM_EVENTS_PATH:
        try:
            os.makedirs(os.path.dirname(LLM_EVENTS_PATH), exist_ok=True)

            # Opening in "w" mode creates a fresh event file for this run.
            # If the file already exists, its previous contents are cleared.
            with open(LLM_EVENTS_PATH, "w", encoding="utf-8"):
                pass

        # Logging should never stop the main pipeline.
        except Exception:
            pass


def _approx_token_count(text):
    """
    Estimate the number of tokens in a piece of text.

    This is only a rough approximation and is used when the LLM/API
    does not provide actual token-usage information.
    """
    if not text:
        return 0

    # Rough assumption:
    # approximately 4 characters ≈ 1 token.
    # max(1, ...) ensures non-empty text is counted as at least 1 token.
    return max(1, int(len(text) / 4))


def log_llm_event(model_name, prompt, response, usage=None):
    """
    Record information about one LLM call.

    One JSON record is appended to llm_events.jsonl for every LLM request.
    The record contains the model name, prompt/response lengths, and
    prompt/completion token counts.
    """

    # If logging has not been initialized, there is nowhere to store
    # the event, so simply return.
    if not LLM_EVENTS_PATH:
        return

    # Start with zero in case token usage is not provided by the LLM.
    prompt_tokens = 0
    completion_tokens = 0

    # Try to use the actual token counts returned by the LLM API.
    if usage is not None:
        try:
            # Some APIs return usage information as a dictionary.
            if isinstance(usage, dict):
                prompt_tokens = int(
                    usage.get("prompt_tokens", 0) or 0
                )
                completion_tokens = int(
                    usage.get("completion_tokens", 0) or 0
                )

            # Other APIs return usage as an object with attributes such as
            # usage.prompt_tokens and usage.completion_tokens.
            else:
                prompt_tokens = int(
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                completion_tokens = int(
                    getattr(usage, "completion_tokens", 0) or 0
                )

        except Exception:
            # If the usage information cannot be read correctly,
            # fall back to token estimation below.
            prompt_tokens = 0
            completion_tokens = 0

    # If the API did not provide a prompt token count, estimate it.
    if prompt_tokens == 0:
        prompt_tokens = _approx_token_count(prompt)

    # If the API did not provide a completion token count, estimate it.
    if completion_tokens == 0:
        completion_tokens = _approx_token_count(response)

    try:
        # Store only useful metrics for this LLM call.
        # The actual full prompt and response are NOT stored here.
        record = {
            "model": str(model_name),
            "prompt_chars": len(prompt or ""),
            "response_chars": len(response or ""),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }

        # Append the record as one JSON object on a new line.
        # "a" means append, so previous LLM-call records are preserved.
        with open(LLM_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # A logging failure should not stop the main data-cleaning pipeline.
    except Exception:
        pass


def write_llm_summary(output_path=None):
    """
    Create the final llm_logging.txt summary from llm_events.jsonl.

    This function:
      1. Reads every recorded LLM call.
      2. Calculates total calls and token usage.
      3. Creates a human-readable summary.
      4. Deletes the temporary llm_events.jsonl file.
    """

    # Use an explicitly provided output path if available.
    # Otherwise, use the path configured earlier by set_llm_log_path().
    target_file = output_path or LLM_LOG_PATH

    # If no output location is known, there is nothing to write.
    if not target_file:
        return

    # Variables used to calculate overall LLM usage.
    total_calls = 0
    total_pt = 0  # Total prompt tokens
    total_ct = 0  # Total completion/output tokens

    # Store the details of every individual LLM call.
    details = []

    # The temporary JSONL file contains one record per LLM call.
    source_file = LLM_EVENTS_PATH

    if source_file and os.path.exists(source_file):
        try:
            # Read the temporary event file one line at a time.
            with open(source_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Convert the JSON line back into a Python dictionary.
                        data = json.loads(line)

                        # Count this LLM call.
                        total_calls += 1

                        # Add its prompt-token usage to the overall total.
                        total_pt += int(
                            data.get("prompt_tokens", 0) or 0
                        )

                        # Add its completion-token usage to the overall total.
                        total_ct += int(
                            data.get("completion_tokens", 0) or 0
                        )

                        # Keep the complete per-call record so it can later
                        # be included in the human-readable report.
                        details.append(data)

                    # Skip a malformed JSON line and continue reading
                    # the remaining LLM-call records.
                    except Exception:
                        continue

        # If the event file itself cannot be read, stop summary generation.
        except Exception:
            return

    # Build the final human-readable report line by line.
    lines = []

    lines.append("LLM Token Logging Summary")
    lines.append(f"total_calls: {total_calls}")
    lines.append(f"total_prompt_tokens: {total_pt}")
    lines.append(f"total_completion_tokens: {total_ct}")

    # Total tokens = prompt tokens + completion tokens.
    lines.append(f"total_tokens: {total_pt + total_ct}")

    # Add a blank line for readability.
    lines.append("")

    lines.append("Details (per call, truncated metrics):")

    # enumerate(..., start=1) makes the call numbering begin at 1
    # instead of Python's default 0.
    for i, d in enumerate(details, start=1):

        # Create one readable line describing this particular LLM call.
        lines.append(
            f"#{i} "
            f"model={d.get('model')} "
            f"prompt_chars={d.get('prompt_chars')} "
            f"response_chars={d.get('response_chars')} "
            f"prompt_tokens={d.get('prompt_tokens')} "
            f"completion_tokens={d.get('completion_tokens')}"
        )

    try:
        # Write the complete report to llm_logging.txt (or the path supplied
        # through output_path).
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # Again, logging/reporting errors should not crash the main pipeline.
    except Exception:
        pass

    try:
        # The JSONL event file is only an intermediate file.
        # Once the final human-readable summary has been written,
        # remove the temporary file.
        if source_file and os.path.exists(source_file):
            os.remove(source_file)

    except Exception:
        pass