from dataclasses import dataclass, field
from pprint import pformat
from pyexpat.errors import messages
from typing import Callable, List, Dict, Tuple, TypeAlias

from src import data


@dataclass
class Message:
	role: str
	content: str

	def as_native(self) -> Dict[str, str]:
		return {"role": self.role, "content": self.content}

	@staticmethod
	def from_native(native: Dict[str, str]) -> "Message":
		return Message(role=native["role"], content=native["content"])

	def __repr__(self) -> str:
		return pformat({"role": self.role, "content": self.content})


@dataclass
class TaggedMessage:
	message: Message
	tag: str

	def as_native(self) -> Tuple[Dict[str, str], str]:
		return (
			{
				"role": self.message.role,
				"content": self.message.content,
			},
			self.tag,
		)

	def as_message(self) -> Message:
		return self.message

	@staticmethod
	def from_native(native: Tuple[Dict[str, str], str]) -> "TaggedMessage":
		return TaggedMessage(
			message=Message(role=native[0]["role"], content=native[0]["content"]),
			tag=native[1],
		)

	def __repr__(self) -> str:
		return (
			"TaggedMessage("
			f"\n\tmessage={self.message.__repr__()}, "
			f"\n\ttag={self.tag}"
			"\n)"
		)


# Example :
# plist = [
#   {"role": "system": "content": "..."},
#   {"role": "user": "content": "..."},
#   {"role": "assistant": "content": "..."},
#   {"role": "user": "content": "..."},
#   {"role": "assistant": "content": "..."},
# ]


@dataclass
class PList:
	messages: List[Message] = field(default_factory=list)

	def __len__(self) -> int:
		return len(self.messages)

	def __add__(self, other: "PList") -> "PList":
		return PList(messages=self.messages + other.messages)

	def as_native(self) -> List[Dict[str, str]]:
		return [message.as_native() for message in self.messages]

	@staticmethod
	def from_native(native: List[Dict[str, str]]) -> "PList":
		return PList(messages=[Message.from_native(message) for message in native])

	def __repr__(self) -> str:
		messages_repr = pformat(self.messages)
		return "PList(" f"\n\tmessages=[\n\t\t" f"{messages_repr}" "\n\t\t]" "\n)"


# Example :
# tagged_plist = [
#   ({"role": "system": "content": "..."}, "system"),
#   ({"role": "user": "content": "..."}, "strategy_gen"),
#   ({"role": "assistant": "content": "..."}, "strategy_reply"),
#   ({"role": "user": "content": "..."}, "code_strategy_gen"),
#   ({"role": "assistant": "content": "..."}, "code_strategy_reply"),
# ]


@dataclass
class TaggedPList:
	messages: List[TaggedMessage] = field(default_factory=list)

	def __len__(self) -> int:
		return len(self.messages)

	def __add__(self, other: "TaggedPList") -> "TaggedPList":
		return TaggedPList(messages=self.messages + other.messages)

	def as_native(self) -> List[Tuple[Dict[str, str], str]]:
		return [message.as_native() for message in self.messages]

	def as_plist(self) -> PList:
		return PList(messages=[message.as_message() for message in self.messages])

	@staticmethod
	def from_native(native: List[Tuple[Dict[str, str], str]]) -> "TaggedPList":
		return TaggedPList(
			messages=[TaggedMessage.from_native(message) for message in native]
		)

	def modify_message_at_index(
		self, index: int, new_tagged_message: TaggedMessage
	) -> "TaggedPList":
		self.messages[index].message = new_tagged_message.message

		return self

	def modify_tag_at_index(self, index: int, new_tag: str) -> "TaggedPList":
		self.messages[index].tag = new_tag

		return self

	def get_tags(self) -> List[str]:
		return [message.tag for message in self.messages]

	def __repr__(self) -> str:
		messages_repr = pformat(self.messages)
		return "TaggedPList(" f"\n\tmessages=[\n\t\t" f"{messages_repr}" "\n\t\t]" "\n)"


GennerType: TypeAlias = Callable[[PList], str]
