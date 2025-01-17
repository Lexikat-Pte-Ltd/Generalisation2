import json
from pathlib import Path
import shutil
import zipfile
import glob


def create_dpo_dataset(json_data):
  dpo_pairs = []

  # Process main_strat_agent first
  main_agent = json_data["main_strat_agent"]
  if main_agent.get("tagged_chat_history"):
    process_agent_history(main_agent, dpo_pairs)

  # Then process copy_strat_agents
  for agent in json_data.get("copy_strat_agents", []):
    process_agent_history(agent, dpo_pairs)

  print(f"Created {len(dpo_pairs)} DPO pairs")
  if dpo_pairs:
    print("\nExample pair:")
    print(json.dumps(dpo_pairs[0], indent=2))

  return dpo_pairs


def process_agent_history(agent, dpo_pairs):
  strategy = agent.get("chosen_strat", "")
  if not strategy:
    return

  context = {
    "strategy": strategy,
    "environment": agent["init_sp_env_info_history"][0][1]
    if agent["init_sp_env_info_history"]
    else "",
  }

  code_attempts = []

  # Process chat history to extract code attempts
  for messages in agent.get("tagged_chat_history", []):
    for msg in messages:
      if isinstance(msg, dict) and msg.get("role") == "assistant":
        content = msg.get("content", "")
        if "```python" in content:
          code = content.split("```python")[1].split("```")[0].strip()
          if code and code not in ("", "\n", "\n\n"):
            code_attempts.append(
              {
                "code": code,
                "tag": msg.get("tag", ""),
                "is_failure": any(
                  x in msg.get("tag", "")
                  for x in ["deletion_fail", "container_fail", "ast_fail"]
                ),
              }
            )

  # Create DPO pairs if we have multiple attempts
  if len(code_attempts) > 1:
    # Use the last non-failure attempt as chosen, or last attempt if all failed
    chosen = next(
      (x for x in reversed(code_attempts) if not x["is_failure"]), code_attempts[-1]
    )

    # Create pairs with rejected attempts
    for rejected in code_attempts:
      if rejected["code"] != chosen["code"]:
        dpo_pairs.append(
          {
            "prompt": f"Task: {strategy}\nEnvironment:\n{context['environment']}",
            "chosen": chosen["code"],
            "rejected": rejected["code"],
            "metadata": {
              "strategy": strategy,
              "environment": context["environment"],
              "chosen_tag": chosen["tag"],
              "rejected_tag": rejected["tag"],
            },
          }
        )


path = Path("./data/").glob("*.json")

for file in path:
  try:
    with open(file, "r") as f:
      data = json.load(f)

    dpo_pairs = create_dpo_dataset(data)

    # Save results
    with open(f"./data/dpo_pairs_of_{file.stem}.json", "w") as f:
      json.dump(dpo_pairs, f, indent=2)
  except Exception as e:
    print(f"Error processing {file}: {e}")
    continue

source_dir = Path("./data")
with zipfile.ZipFile("./data/dpo_pairs_zip.zip", "w") as zf:
  for file in source_dir.glob("dpo_pairs_of_*.json"):
    zf.write(file, file.name)  # file.name keeps just the filename
