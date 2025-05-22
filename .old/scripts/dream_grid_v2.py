#!/usr/bin/env python
from __future__ import annotations

import itertools
import json
import math
import os
import re
import time
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Tuple,
    Callable,
    NamedTuple,
)  # Added NamedTuple for DreamConfig example

from src.config import DreamConfig
from src.genner.Dream import DreamGenner
from src.types import Message, PList


# Third-party imports
from loguru import logger  # Assuming loguru is installed

# Environment variable for the model ID, with a default
DEFAULT_MODEL_ID = os.getenv(
    "GEMINI_MODEL", "gemini-1.5-flash-latest"
)  # Updated to a common model

# Type aliases for clarity
Score = float
ParamSet = Dict[str, Any]
ResultEntry = Tuple[ParamSet, Score, float]  # Parameters, Score, TimeTaken

# System prompt for the grading LLM
# It's crucial that this prompt clearly instructs the LLM to return ONLY JSON.
GRADING_SYSTEM_PROMPT = r"""
You are an automated grader. Your sole responsibility is to evaluate a given Python code snippet.

Return ONLY a JSON object in the format {"score": 0.0}.
The score must be a float between 0.0 and 1.0 (two decimal places are acceptable).

Do not add any explanatory text, greetings, or any other content outside of the JSON object.

Grading Rubric for a Python solution to *invert a binary tree*:

1.00 - Perfect:
	- Defines `class TreeNode:` or uses a LeetCode-style signature `Optional[TreeNode]`.
	- Implements `def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]` (or similar valid name and type hints).
	- Correctly handles the `None` (empty tree) base case.
	- Implements the swap logic for left and right children, either recursively or iteratively (using a stack/queue).
	- Returns the root of the inverted tree.
	- Contains no logical errors or significant stylistic issues.

0.70 - 0.99 - Nearly Correct:
	- The core algorithm for inversion is correct.
	- Minor issues might be present, such as:
	- Missing `return root` statement (if applicable to the chosen approach).
	- Incomplete or missing docstrings for base cases or the main function.
	- Small stylistic flaws that don't affect correctness.
	- Slight deviation from the expected function signature if the logic is sound.

0.40 - 0.69 - Partially Correct:
	- Demonstrates understanding of the tree inversion concept.
	- The code is incomplete, contains significant logical errors, or has issues like:
	- Incorrect return value (e.g., returning nothing or a boolean).
	- Flawed recursive calls (e.g., incorrect base case check leading to infinite recursion for some inputs, though not necessarily for all).
	- Major parts of the algorithm are missing.

0.10 - 0.39 - Minimally Relevant:
	- Mentions tree inversion or related concepts.
	- Provides only pseudocode, high-level commentary, or a non-runnable/incomplete snippet.
	- The code provided is far from a working solution.

0.00 - 0.09 - Irrelevant or Incorrect:
	- The provided solution is completely unrelated to inverting a binary tree.
	- The code is fundamentally wrong or nonsensical for the task.

Remember: ONLY output the JSON. Example: {"score": 0.85}
"""

# Determine which Google Generative AI SDK is available
try:
    import google.generativeai as genai_sdk  # Preferred new SDK

    _SDK_NAME = "generativeai"
except ImportError:
    try:
        import google.genai as genai_sdk  # Older SDK

        _SDK_NAME = "genai"
    except ImportError:
        genai_sdk = None
        _SDK_NAME = None
        logger.error(
            "Neither 'google.generativeai' nor 'google.genai' SDK found. Please install one."
        )
        # Depending on the application, you might want to raise an error here or exit.


@lru_cache(maxsize=128)  # Cache results for identical prompts to save API calls
def _send_to_gemini(full_prompt: str, api_key: str) -> str:
    """
    Sends a prompt to the Gemini API and returns the model's text response.
    Uses lru_cache to avoid redundant API calls for the same prompt.

    Args:
		full_prompt: The complete prompt string to send to the model.
		api_key: The Gemini API key.

    Returns:
		The text content of the model's response.

    Raises:
		RuntimeError: If the SDK is not available or API key is missing.
		google.api_core.exceptions.GoogleAPIError: For API-related errors.
    """
    if not genai_sdk:
        raise RuntimeError("Google Generative AI SDK is not installed.")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is missing or not provided."
        )

    try:
        if _SDK_NAME == "generativeai":  # New SDK (google.generativeai)
            genai_sdk.configure(api_key=api_key)
            model = genai_sdk.GenerativeModel(
                DEFAULT_MODEL_ID,
                # System instruction can be set here if model supports it and it's separated
                # system_instruction="You are an automated grader..." (if GRADING_SYSTEM_PROMPT was split)
            )
            # For safety config, if needed:
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # ]
            # response = model.generate_content(full_prompt, safety_settings=safety_settings)
            response = model.generate_content(full_prompt)
            return response.text
        elif (
            _SDK_NAME == "genai"
        ):  # Legacy SDK (google.genai) - This SDK is less common now.
            # This path might need adjustment based on the exact legacy SDK version's API.
            # The original script used client.models.generate_content, which implies a Client object.
            # However, the more common legacy pattern was similar to the new SDK's configure + GenerativeModel.
            # Assuming a pattern similar to the new SDK for broader compatibility if this path is hit.
            genai_sdk.configure(api_key=api_key)
            model = genai_sdk.GenerativeModel(DEFAULT_MODEL_ID)
            response = model.generate_content(full_prompt)
            return response.text
        else:
            # This case should ideally be caught by the initial SDK check
            raise RuntimeError("SDK name is unknown, this should not happen.")

    except (
        Exception
    ) as e:  # Catching a broad exception, specific API errors are better if known
        logger.error(f"Error during Gemini API call with {_SDK_NAME} SDK: {e}")
        # Re-raise or handle more gracefully depending on requirements
        # For example, if it's a google.api_core.exceptions.GoogleAPIError, you might have specific handling
        raise


def grade_invert_tree_reply(assistant_reply: str) -> Score:
    """
    Grades a given assistant's reply for the "invert binary tree" task
    by sending it to a Gemini model configured with a grading rubric.

    Args:
		assistant_reply: The Python code string generated by the assistant.

    Returns:
		A float score between 0.0 and 1.0.

    Raises:
		ValueError: If the Gemini model returns an unexpected payload or JSON parsing fails.
		RuntimeError: If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable must be set for grading."
        )

    # Construct the full prompt for the grading model
    user_task_description = (
        "Write a Python function `invert_tree(root)` that inverts a binary tree "
        "and returns the new root. Include any helper `TreeNode` class if needed."
    )
    full_prompt_for_grader = (
        f"{GRADING_SYSTEM_PROMPT}\n\n"
        f"USER REQUEST (for context, not for you to answer):\n{user_task_description}\n\n"
        f"ASSISTANT'S PYTHON REPLY (to be graded):\n```python\n{assistant_reply}\n```\n\n"
        "JSON Score:"  # Added a hint for the grader
    )

    raw_response = _send_to_gemini(full_prompt_for_grader, api_key)

    # Attempt to parse the entire response as JSON first
    extracted_json_str = ""
    parsed_json = {}
    try:
        # Strip leading/trailing whitespace which might interfere with JSON parsing
        parsed_json = json.loads(raw_response.strip())
        score = float(parsed_json["score"])
        if not (0.0 <= score <= 1.0):
            logger.warning(
                f"Score {score} is outside the valid 0-1 range. Clamping might be needed or rubric adjusted."
            )
            # Optionally clamp score: score = max(0.0, min(1.0, score))
        return score
    except json.JSONDecodeError:
        logger.warning(
            f"Direct JSON parsing failed for response: '{raw_response}'. Attempting regex extraction."
        )
        # Fallback to regex if the LLM included any extra text despite instructions
        # This regex is greedy (re.S makes . match newlines)
        match = re.search(r"{\s*\"score\"\s*:\s*[\d\.]+\s*}", raw_response, re.DOTALL)
        if match:
            try:
                extracted_json_str = match.group(0)
                parsed_json = json.loads(extracted_json_str)
                score = float(parsed_json["score"])
                if not (0.0 <= score <= 1.0):
                    logger.warning(
                        f"Score {score} (from regex) is outside valid 0-1 range."
                    )
                return score
            except json.JSONDecodeError:
                raise ValueError(
                    f"Regex extracted a non-JSON string: '{extracted_json_str or ''}'. Original response: '{raw_response}'"
                )
            except KeyError:
                raise ValueError(
                    f"Extracted JSON is missing 'score' key: '{parsed_json}'. Original response: '{raw_response}'"
                )
        else:
            raise ValueError(
                f"Gemini grader returned an unexpected payload (no valid JSON found): '{raw_response}'"
            )
    except KeyError:
        raise ValueError(
            f"Parsed JSON is missing 'score' key: '{parsed_json}'. Original response: '{raw_response}'"
        )
    except TypeError:  # Handles cases where 'score' might not be convertible to float
        raise ValueError(
            f"Score value in JSON is not a valid number: '{parsed_json.get('score')}'. Original response: '{raw_response}'"
        )


def _cfg_update(base_config: DreamConfig, overrides: ParamSet) -> DreamConfig:
    """
    Creates a new DreamConfig instance by overriding specified fields in the base_config.
    This assumes DreamConfig is a NamedTuple or a class with a _replace method or similar constructor.

    Args:
		base_config: The initial DreamConfig object.
		overrides: A dictionary of parameters to update in the base_config.

    Returns:
		A new DreamConfig object with updated parameters.
    """
    # If DreamConfig is a NamedTuple, _replace is available
    if hasattr(base_config, "_asdict") and hasattr(
        base_config, "_replace"
    ):  # Checks for NamedTuple attributes
        valid_overrides = {
            k: v for k, v in overrides.items() if k in base_config._fields
        }
        return base_config._replace(**valid_overrides)
    # If DreamConfig is a simple class, you might need a different approach:
    # current_params = {f: getattr(base_config, f) for f in dir(base_config) if not f.startswith('_') and not callable(getattr(base_config,f))}
    # current_params.update({k: v for k, v in overrides.items() if k in current_params})
    # return type(base_config)(**current_params)
    # The original implementation is fine if DreamConfig is a class and hasattr works as expected for its fields.
    else:  # Fallback to original logic, assuming fields are attributes
        config_dict = {
            k: getattr(base_config, k)
            for k in dir(base_config)
            if not k.startswith("_") and not callable(getattr(base_config, k))
        }

        # Filter overrides to only include actual fields of DreamConfig
        # This version is safer if DreamConfig is a regular class
        filtered_overrides = {}
        for k, v in overrides.items():
            if hasattr(base_config, k):
                filtered_overrides[k] = v
            else:
                logger.warning(
                    f"Override key '{k}' not found in DreamConfig, skipping."
                )

        config_dict.update(filtered_overrides)
        return type(base_config)(**config_dict)


def _float_range(start: float, stop: float, step: float) -> List[float]:
    """
    Generates a list of floats from start to stop (inclusive of start,
    may be inclusive or exclusive of stop depending on float precision issues)
    with a given step. Rounds to 10 decimal places to mitigate precision issues.

    Args:
		start: The starting value of the range.
		stop: The ending value of the range.
		step: The step size.

    Returns:
		A list of floats.
    """
    if step == 0:
        return [start] if start <= stop else []
    num_steps = int(math.floor((stop - start) / step)) + 1
    return [
        round(start + i * step, 10)
        for i in range(num_steps)
        if (start + i * step) <= stop + (step / 1000)
    ]  # ensure stop is included


def make_parameter_space(granularity: str = "medium") -> Dict[str, Iterable[Any]]:
    """
    Defines the hyperparameter search space based on the desired granularity.

    Args:
		granularity: A string indicating the coarseness of the search space.
		Expected values: "coarse", "medium", "fine".

    Returns:
		A dictionary where keys are parameter names and values are iterables
		of a_parameter_values to test.

    Raises:
		ValueError: If an unsupported granularity level is provided.
    """
    if granularity not in {"coarse", "medium", "fine"}:
        raise ValueError(
            f"Unsupported granularity: {granularity}. Choose from 'coarse', 'medium', 'fine'."
        )

    space = {
        "max_new_tokens": [64, 128, 256]
        if granularity == "coarse"
        else [32, 64, 96, 128, 160, 192, 224, 256],
        "temperature": (
            _float_range(0.0, 0.5, 0.25)
            if granularity == "coarse"
            else _float_range(0.0, 1.0, 0.25 if granularity == "medium" else 0.05)
        ),
        "top_p": (
            _float_range(0.8, 1.0, 0.2)
            if granularity == "coarse"
            else _float_range(0.5, 1.0, 0.15 if granularity == "medium" else 0.05)
        ),
        # Assuming 'steps' is a parameter for your DreamGenner, e.g., diffusion steps
        "steps": [32, 64, 128]
        if granularity == "coarse"
        else [16, 32, 64, 96, 128, 160, 192],
        "top_k": [0, 20, 40] if granularity == "coarse" else [0, 10, 20, 30, 40, 50],
        "alg": ["entropy", "origin"],  # Algorithm choice for DreamGenner
        "alg_temp": (  # Temperature for the algorithm
            _float_range(0.0, 0.3, 0.3)
            if granularity == "coarse"
            else _float_range(0.0, 0.5, 0.25 if granularity == "medium" else 0.05)
        ),
        "delay": [0.0],  # Delay parameter, perhaps for rate limiting or simulating work
    }
    # Ensure all ranges are correctly generated, especially for single-point "coarse" ranges
    for key, values in space.items():
        if not values:  # If a range accidentally becomes empty
            logger.warning(
                f"Parameter space for '{key}' with granularity '{granularity}' is empty. Check _float_range logic or definitions."
            )
            if key in [
                "temperature",
                "top_p",
                "alg_temp",
            ]:  # Default to a single sensible value
                space[key] = (
                    [0.5]
                    if key == "temperature"
                    else ([0.9] if key == "top_p" else [0.1])
                )

    return space


def grid_search(
    messages_prompt: PList,
    parameter_space: Dict[str, Iterable[Any]],
    objective_function: Callable[[str], Score],
    *,
    base_config: DreamConfig | None = None,
    top_k_results: int = 5,
    verbose: bool = False,
) -> List[ResultEntry]:
    """
    Performs a grid search over the defined parameter_space to find the best
    hyperparameters for the DreamGenner based on the objective_function.

    Args:
		messages_prompt: The initial list of messages to prompt the DreamGenner.
		parameter_space: A dictionary defining the hyperparameters and their values to search.
		objective_function: A callable that takes the generated reply string and returns a score.
		base_config: An optional base DreamConfig to use for default values.
		top_k_results: The number of top-scoring results to return.
		verbose: If True, logs debug information for each combination.

    Returns:
		A list of the top_k_results entries, where each entry is a tuple
		(parameter_set, score, time_taken), sorted by score (descending)
		and then by time_taken (ascending).
    """
    base_config = (
        base_config or DreamConfig()
    )  # Use default DreamConfig if none provided

    param_names = list(parameter_space.keys())
    value_combinations = list(
        itertools.product(*(parameter_space[name] for name in param_names))
    )

    if not value_combinations:
        logger.error(
            "Parameter space resulted in zero combinations. Check make_parameter_space()."
        )
        return []

    logger.info(
        f"Starting grid search. Evaluating {len(value_combinations)} parameter combinations."
    )

    ranked_results: List[ResultEntry] = []

    for idx, combo_values in enumerate(value_combinations, 1):
        current_param_set = dict(zip(param_names, combo_values))
        current_config = _cfg_update(base_config, current_param_set)

        # Ensure DreamGenner is re-initialized with the current_config for each trial
        # This is crucial if DreamGenner's behavior depends on its initial config state.
        # If DreamGenner can have its config updated dynamically, that's an alternative.
        generator = DreamGenner(current_config)

        start_time = time.perf_counter()
        try:
            # Assuming plist_completion takes the list of messages
            reply_obj = generator.plist_completion(messages_prompt)
            # Assuming the reply object has an unwrap() method to get the string
            reply_text = reply_obj.unwrap()

            score = objective_function(reply_text)
            time_taken = time.perf_counter() - start_time

            ranked_results.append((current_param_set, score, time_taken))

            if verbose:
                logger.debug(
                    f"[{idx}/{len(value_combinations)}] Score={score:.3f}, Time={time_taken:.2f}s, Params={current_param_set}"
                )
        except Exception as e:
            # Log the error along with the parameters that caused it
            logger.exception(f"Combination {current_param_set} (idx {idx}) failed: {e}")
            # Optionally, append a failure result (e.g., with score -1 or NaN)
            # ranked_results.append((current_param_set, float('-inf'), time.perf_counter() - start_time))

    # Sort results: highest score first, then by shortest time for ties
    ranked_results.sort(key=lambda x: (-x[1], x[2]))

    return ranked_results[:top_k_results]


if __name__ == "__main__":
    # Configure logging
    # loguru's default stderr logger is fine for scripts.
    # You can customize it, e.g., logger.add("file_{time}.log", level="INFO")

    # Initial conversation prompt for the task
    conversation_prompt = PList(
        [
            Message(
                role="user",
                content="Write a Python function `invert_tree(root)` that inverts a binary tree "
                "and returns the new root. Include any helper TreeNode class if needed.",
            )
        ]
    )

    # Define the granularity of the search: "coarse", "medium", or "fine"
    # Coarse is faster but less thorough. Fine is slower but more detailed.
    search_granularity = "coarse"  # or "medium" or "fine"

    logger.info(
        f"Using GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not Set'}"
    )
    logger.info(f"Using AI Model: {DEFAULT_MODEL_ID}")
    logger.info(f"Using SDK: {_SDK_NAME if _SDK_NAME else 'Unavailable'}")

    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set. The script cannot grade responses.")
        logger.error(
            "Please set the GEMINI_API_KEY environment variable and try again."
        )
        # Example: export GEMINI_API_KEY="your_api_key_here"
    elif not genai_sdk:
        logger.error("Google Generative AI SDK is not available. Please install it.")
        # Example: pip install google-generativeai
    else:
        try:
            param_space = make_parameter_space(search_granularity)

            logger.info(f"Parameter space for '{search_granularity}' search:")
            for name, values in param_space.items():
                logger.info(f"  {name}: {list(values)}")

            top_results = grid_search(
                messages_prompt=conversation_prompt,
                parameter_space=param_space,
                objective_function=grade_invert_tree_reply,
                base_config=DreamConfig(),  # Using default config as base
                top_k_results=10,
                verbose=True,
            )

            print("\n--- Top Hyperparameter Tuning Results ---")
            print(
                f"(Scoring based on Gemini model '{DEFAULT_MODEL_ID}' for 'invert binary tree' task)"
            )
            if top_results:
                for i, (params, score, duration) in enumerate(top_results, 1):
                    print(f"\nRank {i}:")
                    print(f"  Score: {score:.3f}")
                    print(f"  Time Taken: {duration:.2f}s")
                    print(f"  Parameters: {json.dumps(params, indent=2)}")
            else:
                print(
                    "No results found. This might be due to errors during the search or an empty parameter space."
                )

        except Exception as e:
            logger.exception(f"An error occurred during the main execution: {e}")
