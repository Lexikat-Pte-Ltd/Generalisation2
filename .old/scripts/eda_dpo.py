import json
import os
import glob
from collections import Counter, defaultdict
import polars as pl
from typing import Dict, Any
import argparse # Add argparse

# --- Configuration ---
OUTPUT_EDA_FOLDER = "output_eda"
EXPECTED_TOP_KEYS = {
    "main_strat_agent",
    "copy_strat_agents",
    "env_agent",
    "space_freed",
}  # Add other top-level keys if expected


# --- Helper Functions ---
def get_nested_value(data_dict, path, default=None):
    keys = path.split(".")
    val = data_dict
    try:
        for key in keys:
            if isinstance(
                val, list
            ):  # Handle cases where path might involve list indices (not used here but good practice)
                key = int(key)
            val = val[key]
        return val
    except (TypeError, KeyError, IndexError, ValueError):
        return default


def analyze_tagged_history(history: list, file_path: str, agent_path: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "num_turns": 0,
        "role_counts": Counter(),
        "tag_counts": Counter(),
        "empty_content_turns": 0,
        "malformed_turns": 0,
        "consecutive_user_turns": 0,
        "consecutive_assistant_turns": 0,
        "issues": [],
    }
    if not isinstance(history, list):
        stats["issues"].append(f"History is not a list, but {type(history)}")
        return stats

    stats["num_turns"] = len(history)
    last_role = None
    for i, turn_entry in enumerate(history):
        if not isinstance(turn_entry, list) or len(turn_entry) != 2:
            stats["malformed_turns"] += 1
            stats["issues"].append(f"Turn {i} is malformed: Not a list of 2 elements.")
            continue

        turn_data, tag = turn_entry
        if (
            not isinstance(turn_data, dict)
            or "role" not in turn_data
            or "content" not in turn_data
        ):
            stats["malformed_turns"] += 1
            stats["issues"].append(
                f"Turn {i} data is malformed: Missing 'role' or 'content'."
            )
            continue

        role = turn_data["role"]
        content = turn_data["content"]

        stats["role_counts"][role] += 1
        stats["tag_counts"][tag] += 1

        if (
            not content and content != ""
        ):  # Allows empty string but not None or other falsey
            stats["empty_content_turns"] += 1
            stats["issues"].append(
                f"Turn {i} (role: {role}, tag: {tag}) has empty/None content."
            )

        if last_role:
            if role == "user" and last_role == "user":
                stats["consecutive_user_turns"] += 1
            elif role == "assistant" and last_role == "assistant":
                stats["consecutive_assistant_turns"] += 1
        last_role = role

    if (
        stats["num_turns"] < 2 and stats["num_turns"] > 0
    ):  # E.g. only a system prompt or one user message
        stats["issues"].append("Very short history (<2 turns).")
    if stats["malformed_turns"] > 0:
        stats["issues"].append(f"{stats['malformed_turns']} malformed turns detected.")

    return stats


# --- Main EDA Logic ---
def perform_eda_on_folder(input_path: str) -> pl.DataFrame: # Renamed for clarity
    all_file_reports = []
    global_tag_counter = Counter()
    global_role_counter = Counter()
    total_files_processed = 0
    files_with_errors = []
    total_main_agents = 0
    total_copy_agents = 0

    # Construct the glob pattern to find files starting with "full_" and ending with ".json", recursively
    glob_pattern = os.path.join(input_path, "**/full_agent*.json")
    json_files = glob.glob(glob_pattern, recursive=True)
    if not json_files:
        print(f"No JSON files starting with 'full_' found in {input_path} (and its subdirectories using pattern {glob_pattern})")
        return pl.DataFrame()

    print(f"Found {len(json_files)} JSON files to analyze (matching 'full_*.json' in {input_path})...")

    for file_idx, file_path in enumerate(json_files):
        report = {
            "file_path": file_path,
            "issues": [],
            "main_agent_issues": [],
            "copy_agents_issues": [],
        }
        print(f"\nAnalyzing file ({file_idx + 1}/{len(json_files)}): {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_files_processed += 1
        except json.JSONDecodeError as e:
            report["issues"].append(f"JSONDecodeError: {e}")
            files_with_errors.append({"file": file_path, "error": str(e)})
            all_file_reports.append(report)
            continue
        except Exception as e:
            report["issues"].append(f"FileReadError: {e}")
            files_with_errors.append({"file": file_path, "error": str(e)})
            all_file_reports.append(report)
            continue

        print(len(data))

        # Top-level key checks
        missing_top_keys = EXPECTED_TOP_KEYS - set(data.keys())
        if missing_top_keys:
            report["issues"].append(f"Missing top-level keys: {missing_top_keys}")

        report["file_size_kb"] = os.path.getsize(file_path) / 1024
        report["total_top_level_keys"] = len(data.keys())

        # Main Strat Agent Analysis
        main_agent_data = get_nested_value(data, "main_strat_agent")
        if isinstance(main_agent_data, dict):
            total_main_agents += 1
            report["main_agent_present"] = True
            main_strats = get_nested_value(main_agent_data, "strats", [])
            report["main_agent_num_strats"] = (
                len(main_strats) if isinstance(main_strats, list) else 0
            )
            if not main_strats:
                report["main_agent_issues"].append(
                    "main_strat_agent.strats is empty or missing."
                )

            main_history = get_nested_value(main_agent_data, "tagged_chat_history", [])
            if not main_history:
                report["main_agent_issues"].append(
                    "main_strat_agent.tagged_chat_history is empty or missing."
                )
            elif not isinstance(main_history, list):
                report["main_agent_issues"].append(
                    f"main_strat_agent.tagged_chat_history is not a list, but {type(main_history)}."
                )
            else:
                hist_stats_main = analyze_tagged_history(
                    main_history, file_path, "main_strat_agent"
                )
                report["main_agent_history_stats"] = hist_stats_main
                global_tag_counter.update(hist_stats_main["tag_counts"])
                global_role_counter.update(hist_stats_main["role_counts"])
                if hist_stats_main["issues"]:
                    report["main_agent_issues"].extend(
                        [f"History issue: {iss}" for iss in hist_stats_main["issues"]]
                    )
        else:
            report["main_agent_present"] = False
            report["main_agent_issues"].append(
                "main_strat_agent key is missing or not a dict."
            )

        # Copy Strat Agents Analysis
        copy_agents_data = get_nested_value(data, "copy_strat_agents", [])
        if isinstance(copy_agents_data, list):
            report["num_copy_agents"] = len(copy_agents_data)
            total_copy_agents += len(copy_agents_data)
            agent_reports = []
            for i, copy_agent in enumerate(copy_agents_data):
                agent_path_id = f"copy_strat_agents[{i}]"
                agent_report = {"id": agent_path_id, "issues": []}
                if not isinstance(copy_agent, dict):
                    agent_report["issues"].append("Agent data is not a dict.")
                    agent_reports.append(agent_report)
                    continue

                chosen_strat = get_nested_value(copy_agent, "chosen_strat")
                agent_report["chosen_strat_present"] = bool(chosen_strat)
                if not chosen_strat:
                    agent_report["issues"].append("chosen_strat is missing or empty.")

                # Check if chosen_strat is in main_agent_strats (if main agent & strats exist)
                main_strats_list = get_nested_value(data, "main_strat_agent.strats", [])
                if (
                    chosen_strat
                    and isinstance(main_strats_list, list)
                    and main_strats_list
                    and chosen_strat not in main_strats_list
                ):
                    agent_report["issues"].append(
                        f"chosen_strat '{chosen_strat[:30]}...' not found in main_strat_agent.strats."
                    )

                agent_report["space_freed"] = get_nested_value(
                    copy_agent, "space_freed"
                )
                if agent_report["space_freed"] is None:
                    agent_report["issues"].append("space_freed is missing.")
                elif not isinstance(agent_report["space_freed"], (int, float)):
                    agent_report["issues"].append(
                        f"space_freed is not a number: {type(agent_report['space_freed'])}."
                    )

                copy_history = get_nested_value(copy_agent, "tagged_chat_history", [])
                hist_stats_copy: Dict[str, Any] = {"tag_counts": Counter(), "issues": []} # Initialize here

                if not copy_history:
                    agent_report["issues"].append(
                        "tagged_chat_history is empty or missing."
                    )
                elif not isinstance(copy_history, list):
                    agent_report["issues"].append(
                        f"tagged_chat_history is not a list, but {type(copy_history)}."
                    )
                else:
                    hist_stats_copy = analyze_tagged_history(
                        copy_history, file_path, agent_path_id
                    )
                    agent_report["history_stats"] = hist_stats_copy
                    global_tag_counter.update(hist_stats_copy["tag_counts"])
                    global_role_counter.update(hist_stats_copy["role_counts"])
                    if hist_stats_copy["issues"]:
                        agent_report["issues"].extend(
                            [f"History issue: {iss}" for iss in hist_stats_copy["issues"]]
                        )

                # Consistency check: space_freed vs code generation tags
                if (
                    isinstance(agent_report["space_freed"], (int, float))
                    and agent_report["space_freed"] <= 0
                    and any(
                        "genned_strat_code(success)" in tag
                        for tag in hist_stats_copy.get("tag_counts", {})
                    )
                ):
                    agent_report["issues"].append(
                        "Inconsistency: space_freed <= 0 but a 'genned_strat_code(success)' tag exists."
                    )

                if agent_report[
                    "issues"
                ]:  # If this specific copy agent has issues, add to file's copy_agent_issues
                    report["copy_agents_issues"].append(
                        f"Agent {i}: {'; '.join(agent_report['issues'])}"
                    )
                agent_reports.append(agent_report)
            report["copy_agent_details"] = (
                agent_reports  # Store all details for potential deeper dive
            )
        else:
            report["num_copy_agents"] = 0
            report["copy_agents_issues"].append(
                "copy_strat_agents key is missing or not a list."
            )

        # Combine all issues for a summary
        combined_issues = (
            report["issues"]
            + report["main_agent_issues"]
            + report["copy_agents_issues"]
        )
        report["has_any_issues"] = bool(combined_issues)
        report["issue_summary_top3"] = "; ".join(combined_issues[:3])

        all_file_reports.append(report)

    # --- Generate Summary Report ---
    os.makedirs(OUTPUT_EDA_FOLDER, exist_ok=True)

    print("\n\n--- EDA Summary Report ---")
    print(f"Total JSON files found: {len(json_files)}")
    print(f"Total JSON files successfully parsed: {total_files_processed}")
    if files_with_errors:
        print(f"Files with parsing/read errors: {len(files_with_errors)}")
        for err_file in files_with_errors:
            print(f"  - {err_file['file']}: {err_file['error']}")
        with open(
            os.path.join(OUTPUT_EDA_FOLDER, "files_with_parsing_errors.json"), "w"
        ) as f:
            json.dump(files_with_errors, f, indent=2)

    print(f"\nTotal main_strat_agents found: {total_main_agents}")
    print(f"Total copy_strat_agents found: {total_copy_agents}")

    print("\nGlobal Role Counts across all histories:")
    for role, count in global_role_counter.most_common():
        print(f"  - {role}: {count}")

    print("\nGlobal Tag Counts across all histories (Top 20):")
    for tag, count in global_tag_counter.most_common(20):
        print(f"  - {tag}: {count}")
    with open(os.path.join(OUTPUT_EDA_FOLDER, "global_tag_counts.json"), "w") as f:
        json.dump(global_tag_counter, f, indent=2)

    # Convert reports to DataFrame for easier analysis and CSV export
    if not all_file_reports:
        print("No data to report (all files might have failed parsing).")
        return pl.DataFrame()

    try:
        df_report = pl.DataFrame(all_file_reports)
    except Exception as e:
        print(f"Error creating Polars DataFrame: {e}")
        print("Sample data that caused error (first item):")
        if all_file_reports:
            print(json.dumps(all_file_reports[0], indent=2, default=str))
        return pl.DataFrame()


    # Simplify complex columns for CSV, or explode them if needed for detailed analysis
    # For now, we'll keep a summary of issues.
    columns_for_summary = [
        "file_path",
        "file_size_kb",
        "has_any_issues",
        "issue_summary_top3",
        "main_agent_present",
        "main_agent_num_strats",
        "num_copy_agents",
    ]
    # Ensure all selected columns exist in the DataFrame to avoid errors
    existing_columns_for_summary = [col for col in columns_for_summary if col in df_report.columns]
    if not existing_columns_for_summary:
        print(f"Warning: None of the expected summary columns {columns_for_summary} found in DataFrame. Saving an empty summary CSV.")
        df_report_summary = pl.DataFrame()
    else:
        if len(existing_columns_for_summary) != len(columns_for_summary):
            missing_cols = set(columns_for_summary) - set(existing_columns_for_summary)
            print(f"Warning: The following summary columns are missing from the DataFrame and will not be in the CSV: {missing_cols}")
        df_report_summary = df_report.select(existing_columns_for_summary)

    csv_path = os.path.join(OUTPUT_EDA_FOLDER, "eda_summary_report.csv")
    try:
        df_report_summary.write_csv(csv_path)
        print(f"\nDetailed summary report saved to: {csv_path}")
    except Exception as e:
        print(f"Error writing summary CSV: {e}")

    # Save full report details (can be large)
    full_report_path = os.path.join(OUTPUT_EDA_FOLDER, "eda_full_details.json")

    # Need a custom encoder for Counter objects if they are still in the dicts
    # For simplicity, let's ensure they are converted or removed for full JSON dump
    def prepare_for_json(obj: Any) -> Any:
        if isinstance(obj, Counter):
            return dict(obj)
        # Polars data types are generally fine with json.dumps or are converted to Python natives
        # No specific Polars timestamp handling needed here like for pd.Timestamp
        return obj

    try:
        with open(full_report_path, "w") as f:
            # Create a serializable version of all_file_reports
            serializable_reports = []
            for r_dict in all_file_reports:
                sr = {}
                for k, v in r_dict.items():
                    if k == "main_agent_history_stats" or k == "copy_agent_details":
                        sr[k] = json.loads(
                            json.dumps(v, default=prepare_for_json)
                        )  # Force serialization
                    else:
                        sr[k] = v
                serializable_reports.append(sr)
            json.dump(
                serializable_reports, f, indent=2, default=str
            )  # Use str as fallback
        print(f"Full detailed report (JSON) saved to: {full_report_path}")
    except Exception as e:
        print(
            f"Error saving full JSON report: {e}. Some data might not be serializable."
        )

    # Identify files with specific critical issues for exclusion
    if "has_any_issues" in df_report.columns and "file_path" in df_report.columns:
        files_to_potentially_exclude = (
            df_report.filter(pl.col("has_any_issues") == True)
            .select("file_path")
            .to_series()
            .to_list()
        )
    else:
        files_to_potentially_exclude = []
        print("Warning: 'has_any_issues' or 'file_path' column not found in df_report. Cannot generate exclusion list.")

    if files_to_potentially_exclude:
        print(
            f"\nFound {len(files_to_potentially_exclude)} files with one or more identified issues."
        )
        print(
            "Consider reviewing these files (details in eda_full_details.json and summary in eda_summary_report.csv):"
        )
        # for f_path in files_to_potentially_exclude[:10]: # Print first 10
        #     print(f"  - {f_path}")
        exclusion_list_path = os.path.join(
            OUTPUT_EDA_FOLDER, "potential_exclusion_list.txt"
        )
        with open(exclusion_list_path, "w") as f:
            for line in files_to_potentially_exclude:
                f.write(f"{line}\n")
        print(f"List of files with issues saved to: {exclusion_list_path}")
    else:
        print("\nNo files flagged with major issues based on current checks.")

    return df_report_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on JSON files starting with 'full_' in a specified folder.")
    parser.add_argument(
        "data_folder",
        type=str,
        help="Path to the folder to scan for JSON files (e.g., ./agent_data). Will recursively search for 'full_*.json'.",
    )
    args = parser.parse_args()
    data_folder_to_scan = args.data_folder

    if not os.path.isdir(data_folder_to_scan):
        print(f"Error: Folder '{data_folder_to_scan}' not found.")
    else:
        print(f"Starting EDA for folder: {data_folder_to_scan}, looking for 'full_*.json' files.")
        summary_df = perform_eda_on_folder(data_folder_to_scan)
        if summary_df.height > 0:
            print("\n--- EDA Process Finished ---")
            print("Review the generated files in the 'output_eda' folder.")
        else:
            print("\n--- EDA Process Finished (No data or files processed) ---")
