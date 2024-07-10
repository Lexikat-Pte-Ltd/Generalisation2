from typing import List, Dict, Tuple

# Example :
# message = {"role": "system": "content": "..."}
Message = Dict[str, str]

# Example :
# tagged_message = ({"role": "system": "content": "..."}, "system")
TaggedMessage = Tuple[Message, str]

# Example :
# plist = [
#   {"role": "system": "content": "..."},
#   {"role": "user": "content": "..."},
#   {"role": "assistant": "content": "..."},
#   {"role": "user": "content": "..."},
#   {"role": "assistant": "content": "..."},
# ]
PList = List[Message]

# Example :
# tagged_plist = [
#   ({"role": "system": "content": "..."}, "system"),
#   ({"role": "user": "content": "..."}, "strategy_gen"),
#   ({"role": "assistant": "content": "..."}, "strategy_reply"),
#   ({"role": "user": "content": "..."}, "code_strategy_gen"),
#   ({"role": "assistant": "content": "..."}, "code_strategy_reply"),
# ]
TaggedPList = List[TaggedMessage]


