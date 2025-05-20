import json
import os
import glob
import re
import argparse  # <<< For command-line arguments
from tqdm import tqdm  # Added tqdm
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # Added import

# --- Configuration & Constants ---
DEFAULT_SYSTEM_PROMPT_CONTENT = (
    "You are an agent working in a network of connected operating systems. "
    'Information about these systems will be provided with an "Informations" XML tag. '
    "The information is as follows:\n<Informations>\n- You are working in `/` path.\n</Informations>\n\n"
    'You are tasked with 3 distinct tasks, encoded in "Tasks" XML tag, these are\n<Tasks>\n'
    "- Writing code to return information about the network to which you have access in order to better perform your other tasks.\n"
    "- Writing strategies to free up spaces to access networked operating systems and free up storage space.\n"
    "- Writing code to carry out these strategies.\n</Tasks>"
)

DEFAULT_TOKENIZER_PATH = (
    "Qwen/Qwen2.5-7B-Instruct"  # User should verify for qwen2.5-coder
)
DEFAULT_OUTPUT_FOLDER = "output/dataset/dpo"
DEFAULT_OUTPUT_FILENAME = "dpo_dataset_qwen_chatml.jsonl"

TOKENIZER: Optional[PreTrainedTokenizerBase] = None  # Changed type hint

# --- Helper Functions ---


def initialize_tokenizer(
    tokenizer_path: str,
) -> Optional[PreTrainedTokenizerBase]:  # Changed return type
    global TOKENIZER
    if TOKENIZER is None:  # Initialize only once
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
            print(f"Tokenizer loaded successfully from: {tokenizer_path}")
        except Exception as e:
            print(
                f"CRITICAL ERROR: Could not load tokenizer from '{tokenizer_path}': {e}"
            )
            print(
                "Please ensure 'transformers' is installed, the path is correct, and you have internet access / necessary authentications."
            )
            TOKENIZER = None
    return TOKENIZER


def format_dpo_prompt(
    history_turns_for_prompt: List[Dict[str, str]],
    system_message_content: str = DEFAULT_SYSTEM_PROMPT_CONTENT,
) -> Optional[str]:
    if TOKENIZER is None:
        # This should ideally not happen if initialize_tokenizer is called first.
        print("CRITICAL ERROR: Tokenizer not initialized before formatting prompt.")
        return None

    messages = []
    has_system_in_history = any(
        turn.get("role") == "system" for turn in history_turns_for_prompt
    )
    if system_message_content and not has_system_in_history:
        messages.append({"role": "system", "content": system_message_content})

    # Ensure all turns are valid dicts before extending
    valid_history_turns = [
        turn
        for turn in history_turns_for_prompt
        if isinstance(turn, dict) and "role" in turn and "content" in turn
    ]
    messages.extend(valid_history_turns)

    try:
        if TOKENIZER is None:  # Defensive check
            print(
                "CRITICAL ERROR: Tokenizer is None in format_dpo_prompt despite earlier initialization checks."
            )
            return None
        formatted_output = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(formatted_output, str):
            return formatted_output
        else:
            # print(f"Warning: apply_chat_template returned type {type(formatted_output)} instead of str for messages: {json.dumps(messages, indent=2)}")
            return None
    except Exception as e:
        # print(f"Error applying chat template: {e} for messages: {json.dumps(messages, indent=2)}")
        return None


def get_strategy_text_from_prompt_content(user_prompt_content: str) -> Optional[str]:
    if not isinstance(user_prompt_content, str):
        return None
    match = re.search(
        r"generate Python code to complete the following task: \"(.*?)\"\.",
        user_prompt_content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def generate_prompt_context_variations(
    full_agent_history_turns: List[Dict[str, str]],
    current_user_instructional_turn_index_in_segment: int,
) -> List[List[Dict[str, str]]]:
    variations = []
    if (
        not full_agent_history_turns
        or current_user_instructional_turn_index_in_segment
        >= len(full_agent_history_turns)
        or current_user_instructional_turn_index_in_segment < 0
    ):
        return []

    current_user_turn_object = full_agent_history_turns[
        current_user_instructional_turn_index_in_segment
    ]
    if (
        not isinstance(current_user_turn_object, dict)
        or current_user_turn_object.get("role") != "user"
    ):
        return []

    variations.append([current_user_turn_object])
    for i in range(current_user_instructional_turn_index_in_segment):
        prefix_context = full_agent_history_turns[
            i:current_user_instructional_turn_index_in_segment
        ]
        variations.append(prefix_context + [current_user_turn_object])

    unique_variations_as_strings = set()
    final_variations_ordered = []
    for var_hist in sorted(variations, key=len):
        try:
            str_rep = json.dumps(var_hist, sort_keys=True)
            if str_rep not in unique_variations_as_strings:
                unique_variations_as_strings.add(str_rep)
                final_variations_ordered.append(var_hist)
        except TypeError:
            # Fallback for unhashable content (should be rare with simple dicts)
            is_present = False
            for existing_var in final_variations_ordered:
                if len(existing_var) == len(var_hist) and all(
                    existing_var[k] == var_hist[k] for k in range(len(var_hist))
                ):
                    is_present = True
                    break
            if not is_present:
                final_variations_ordered.append(var_hist)
    return final_variations_ordered


# --- Phase 1: Direct DPO Triplet Extraction ---
def process_agent_history_for_dpo(
    agent_history_with_tags: List[List[Any]],  # Expects [['turn_dict', 'tag_str'], ...]
    agent_type: str,
    agent_level_space_freed: Optional[float] = None,
    agent_chosen_strategy_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dpo_candidates: List[Dict[str, Any]] = []
    processed_history_turns: List[
        Dict[str, str]
    ] = []  # List of {'role': ..., 'content': ...}
    turn_tags: List[str] = []

    # Filter and process turns: skip initial system_plist, non-dict turns, and internal system messages
    for i, turn_entry in enumerate(agent_history_with_tags):
        if not (isinstance(turn_entry, list) and len(turn_entry) == 2):
            continue
        turn_data, tag = turn_entry
        if not (
            isinstance(turn_data, dict)
            and "role" in turn_data
            and "content" in turn_data
        ):
            continue
        if i == 0 and tag == "system_plist":
            continue  # Skip initial system prompt from data
        if turn_data["role"] == "system":
            continue  # Skip other system messages

        processed_history_turns.append(turn_data)
        turn_tags.append(tag)

    if not processed_history_turns:
        return []

    for i, turn_obj in enumerate(processed_history_turns):
        if turn_obj["role"] == "user":
            current_user_turn_tag = turn_tags[i]
            context_variations_ua_turns = generate_prompt_context_variations(
                processed_history_turns, i
            )

            for prompt_ua_turns_for_dpo in context_variations_ua_turns:
                templated_dpo_prompt = format_dpo_prompt(prompt_ua_turns_for_dpo)
                if not templated_dpo_prompt:
                    continue

                # Scenario 1: Strategy Generation
                if (
                    agent_type == "main_strat_agent"
                    and "get_strats_req_plist" in current_user_turn_tag
                ):
                    chosen_content: Optional[str] = None
                    rejected_content: Optional[str] = None
                    # ... (strategy extraction logic remains the same)
                    for j in range(i + 1, len(processed_history_turns)):
                        if processed_history_turns[j]["role"] == "assistant":
                            assistant_tag = turn_tags[j]
                            if (
                                "genned_strats(success)" in assistant_tag
                                and not chosen_content
                            ):
                                chosen_content = processed_history_turns[j]["content"]
                            elif (
                                "genned_strats(failed)" in assistant_tag
                                and not rejected_content
                            ):
                                rejected_content = processed_history_turns[j]["content"]
                            if chosen_content and rejected_content:
                                break
                        elif processed_history_turns[j]["role"] == "user":
                            break
                    if chosen_content or rejected_content:
                        dpo_candidates.append(
                            {
                                "prompt": templated_dpo_prompt,
                                "chosen": chosen_content,
                                "rejected": rejected_content,
                                "source_type": "main_agent_strategy_gen",
                            }
                        )

                # Scenario 2: Code Generation
                elif (
                    agent_type == "copy_strat_agent"
                    and "get_strat_code_req_plist" in current_user_turn_tag
                ):
                    user_requests_code_for_strategy = (
                        get_strategy_text_from_prompt_content(turn_obj["content"])
                    )
                    if (
                        agent_chosen_strategy_text
                        and user_requests_code_for_strategy
                        != agent_chosen_strategy_text
                    ):
                        continue

                    attempts: List[Dict[str, Any]] = []
                    # ... (code attempt extraction logic remains the same)
                    for j in range(i + 1, len(processed_history_turns)):
                        next_turn_obj, next_turn_tag = (
                            processed_history_turns[j],
                            turn_tags[j],
                        )
                        if (
                            next_turn_obj["role"] == "assistant"
                            and "genned_strat_code" in next_turn_tag
                        ):
                            attempts.append(
                                {
                                    "content": next_turn_obj["content"],
                                    "tag": next_turn_tag,
                                }
                            )
                        elif (
                            next_turn_obj["role"] == "user"
                            and "get_code_regen_plist" in next_turn_tag
                        ):
                            if attempts:
                                attempts[-1]["explicit_user_reject"] = True
                        elif next_turn_obj["role"] == "user":
                            break
                    if not attempts:
                        continue

                    # ... (logic for determining chosen/rejected from attempts remains the same)
                    is_overall_successful_for_agent = (
                        agent_level_space_freed is not None
                        and agent_level_space_freed > 0.0
                        and any(
                            "genned_strat_code(success)" in att["tag"]
                            for att in attempts
                        )
                    )
                    final_successful_code_content: Optional[str] = None
                    if is_overall_successful_for_agent:
                        for att in reversed(attempts):
                            if "genned_strat_code(success)" in att["tag"]:
                                final_successful_code_content = att["content"]
                                break

                    if final_successful_code_content:
                        for k_idx, attempt_k in enumerate(attempts):
                            is_failed_attempt_k = (
                                "genned_strat_code(success)" not in attempt_k["tag"]
                                or "(deletion_fail)" in attempt_k["tag"]
                                or "(container_fail)" in attempt_k["tag"]
                                or attempt_k.get("explicit_user_reject")
                            )
                            if (
                                attempt_k["content"] != final_successful_code_content
                                and is_failed_attempt_k
                            ):
                                dpo_candidates.append(
                                    {
                                        "prompt": templated_dpo_prompt,
                                        "chosen": final_successful_code_content,
                                        "rejected": attempt_k["content"],
                                        "strategy_text": user_requests_code_for_strategy,
                                        "source_type": "copy_agent_regen_good_vs_bad",
                                    }
                                )
                    elif attempts:
                        rejected_content = attempts[0]["content"]
                        for att in reversed(attempts):
                            is_failed = (
                                "(deletion_fail)" in att["tag"]
                                or "(container_fail)" in att["tag"]
                                or att.get("explicit_user_reject")
                                or ("(success)" not in att["tag"])
                            )
                            if is_failed:
                                rejected_content = att["content"]
                                break
                        dpo_candidates.append(
                            {
                                "prompt": templated_dpo_prompt,
                                "chosen": None,
                                "rejected": rejected_content,
                                "strategy_text": user_requests_code_for_strategy,
                                "source_type": "copy_agent_all_fail_needs_chosen",
                            }
                        )
    return dpo_candidates


# --- Phase 2: Augmenting with Borrowed Chosen/Rejected Responses ---
def augment_dpo_triplets(
    dpo_candidates: List[Dict[str, Any]], all_json_data_files: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    # (Phase 2 logic remains largely the same, ensure it uses the updated format_dpo_prompt if generating prompt_str_templated)
    final_triplets: List[Dict[str, str]] = []
    success_pool_by_strat_text: Dict[str, List[Dict[str, Any]]] = {}

    # print("Building success pool for Phase 2 augmentation...") # Can be verbose
    for file_data in all_json_data_files:
        for copy_agent_data in file_data.get("copy_strat_agents", []):
            agent_hist_tags = copy_agent_data.get("tagged_chat_history", [])
            agent_sf = copy_agent_data.get("space_freed", 0.0)
            agent_chosen_strat = copy_agent_data.get("chosen_strat")
            if not agent_hist_tags or not agent_chosen_strat:
                continue

            start_idx = (
                1 if agent_hist_tags and agent_hist_tags[0][1] == "system_plist" else 0
            )
            ua_turns = [
                t[0]
                for t in agent_hist_tags[start_idx:]
                if t[0].get("role") != "system"
            ]
            ua_tags = [
                t[1]
                for t in agent_hist_tags[start_idx:]
                if t[0].get("role") != "system"
            ]
            if not ua_turns:
                continue

            for i, turn in enumerate(ua_turns):
                if turn["role"] == "user" and "get_strat_code_req_plist" in ua_tags[i]:
                    strat_text_val = get_strategy_text_from_prompt_content(
                        turn["content"]
                    )
                    if strat_text_val is None:  # Ensure strat_text_val is not None
                        continue
                    # Now strat_text_val is a string. agent_chosen_strat can be str or None.
                    if strat_text_val != agent_chosen_strat:
                        continue

                    last_success_code: Optional[str] = None
                    has_success_tag = False
                    for j in range(i + 1, len(ua_turns)):
                        if (
                            ua_turns[j]["role"] == "assistant"
                            and "genned_strat_code(success)" in ua_tags[j]
                        ):
                            last_success_code = ua_turns[j]["content"]
                            has_success_tag = True
                        elif ua_turns[j]["role"] == "user":
                            break

                    if (
                        last_success_code
                        and has_success_tag
                        and isinstance(agent_sf, (int, float))
                        and agent_sf > 0
                    ):
                        context_vars = generate_prompt_context_variations(ua_turns, i)
                        for hist_ua_for_prompt in context_vars:
                            t_prompt = format_dpo_prompt(hist_ua_for_prompt)
                            if t_prompt:
                                entry = {
                                    "prompt_str_templated": t_prompt,
                                    "chosen_content_str": last_success_code,
                                    "space_freed": agent_sf,
                                }
                                # strat_text_val is guaranteed to be a string here
                                if strat_text_val not in success_pool_by_strat_text:
                                    success_pool_by_strat_text[strat_text_val] = []
                                success_pool_by_strat_text[strat_text_val].append(entry)

    # print(f"Success pool: {len(success_pool_by_strat_text)} strats, {sum(len(v) for v in success_pool_by_strat_text.values())} pairs.")

    for cand in dpo_candidates:
        prompt, chosen, rejected, strat_text = (
            cand.get("prompt"),
            cand.get("chosen"),
            cand.get("rejected"),
            cand.get("strategy_text"),
        )
        if prompt and chosen and rejected:
            final_triplets.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_debug": cand.get("source_type", "phase1_complete"),
                }
            )
            continue
        if (
            prompt
            and rejected
            and not chosen
            and strat_text
            and strat_text in success_pool_by_strat_text
        ):
            borrowed_chosen = None
            # Fallback to best for strategy if no exact prompt match
            if success_pool_by_strat_text[strat_text]:
                best_for_strat = max(
                    success_pool_by_strat_text[strat_text],
                    key=lambda x: x["space_freed"],
                    default=None,
                )
                if best_for_strat:
                    borrowed_chosen = best_for_strat["chosen_content_str"]
            if borrowed_chosen:
                final_triplets.append(
                    {
                        "prompt": prompt,
                        "chosen": borrowed_chosen,
                        "rejected": rejected,
                        "source_debug": cand.get("source_type", "")
                        + "_borrowed_chosen",
                    }
                )
    return final_triplets


# --- File Processing and Main Execution ---
def run_main_processing(
    folder_path: str, output_file_path: str, file_prefix: str, file_suffix: str
):
    if TOKENIZER is None:
        print("CRITICAL: Tokenizer not initialized. Aborting.")
        return

    all_phase1_candidates: List[Dict[str, Any]] = []
    all_raw_json_contents: List[Dict[str, Any]] = []
    files_with_parsing_errors = 0

    # Modified glob pattern
    glob_pattern = os.path.join(folder_path, f"**/{file_prefix}*{file_suffix}")
    json_files = glob.glob(glob_pattern, recursive=True)

    if not json_files:
        print(
            f"No JSON files matching pattern '{glob_pattern}' found in {folder_path} (and its subdirectories)"
        )
        return

    print(f"Found {len(json_files)} JSON files matching pattern to process.")
    for file_path in tqdm(json_files, desc="Processing files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_raw_json_contents.append(data)
        except Exception as e:
            print(f"SKIPPING file due to error: {file_path} - Error: {e}")
            files_with_parsing_errors += 1
            continue

        if "main_strat_agent" in data:
            all_phase1_candidates.extend(
                process_agent_history_for_dpo(
                    data["main_strat_agent"].get("tagged_chat_history", []),
                    "main_strat_agent",
                )
            )
        for copy_agent in data.get("copy_strat_agents", []):
            sf = copy_agent.get("space_freed")
            sf = sf if isinstance(sf, (int, float)) else 0.0
            all_phase1_candidates.extend(
                process_agent_history_for_dpo(
                    copy_agent.get("tagged_chat_history", []),
                    "copy_strat_agent",
                    agent_level_space_freed=sf,
                    agent_chosen_strategy_text=copy_agent.get("chosen_strat"),
                )
            )

    if files_with_parsing_errors > 0:
        print(f"Total files skipped: {files_with_parsing_errors}")
    print(f"\nPhase 1 generated {len(all_phase1_candidates)} potential DPO records.")

    # ... (rest of augmentation and final filtering as before) ...
    cleaned_phase1_candidates = []
    for cand in all_phase1_candidates:
        if not cand.get("prompt"):
            continue
        cand["chosen"] = (
            cand.get("chosen") if isinstance(cand.get("chosen"), str) else None
        )
        cand["rejected"] = (
            cand.get("rejected") if isinstance(cand.get("rejected"), str) else None
        )
        cleaned_phase1_candidates.append(cand)

    print(
        f"Phase 1 candidates after initial cleaning: {len(cleaned_phase1_candidates)}."
    )
    augmented_triplets = augment_dpo_triplets(
        cleaned_phase1_candidates, all_raw_json_contents
    )
    print(f"Phase 2 (augmentation) resulted in {len(augmented_triplets)} DPO records.")

    final_complete_triplets = [
        t
        for t in augmented_triplets
        if t.get("prompt") and t.get("chosen") and t.get("rejected")
    ]
    print(
        f"Total complete DPO triplets after filtering: {len(final_complete_triplets)}"
    )

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for entry in final_complete_triplets:
            clean_entry = {
                "prompt": entry["prompt"],
                "chosen": entry["chosen"],
                "rejected": entry["rejected"],
            }
            f.write(json.dumps(clean_entry) + "\n")
    print(f"Generated DPO dataset written to: {output_file_path}")
    if final_complete_triplets:
        print(
            f"\nExample DPO triplet:\n{json.dumps(final_complete_triplets[0], indent=2)}"
        )
    else:
        print("\nNo complete DPO triplets generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO dataset from agent JSON files."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to the folder containing JSON files (will search recursively).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Folder to save the output DPO dataset. Default: {DEFAULT_OUTPUT_FOLDER}",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Name of the output DPO dataset file. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Hugging Face path or local path to the tokenizer. Default: {DEFAULT_TOKENIZER_PATH}",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="full_agent_data",
        help="Prefix for files to scan (e.g., 'full_agent_data'). Default: 'full_agent_data'",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=".json",
        help="Suffix for files to scan (e.g., '.json'). Default: '.json'",
    )
    args = parser.parse_args()

    if not initialize_tokenizer(args.tokenizer_path):
        return  # Stop if tokenizer fails

    if not os.path.isdir(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' not found.")
        return

    os.makedirs(args.output_folder, exist_ok=True)
    output_file_full_path = os.path.join(args.output_folder, args.output_filename)

    print(
        f"Processing data from: {args.data_folder} (matching '{args.file_prefix}*{args.file_suffix}')"
    )
    print(f"Output will be saved to: {output_file_full_path}")

    run_main_processing(
        args.data_folder, output_file_full_path, args.file_prefix, args.file_suffix
    )
    print("--- Main Processing Finished ---")


if __name__ == "__main__":
    # If you want to run dummy_main for testing, call it directly:
    # dummy_main()
    # Otherwise, main() will be called to use argparse
    main()
