from typing import List, Literal, TypedDict, NotRequired

from src.typing.message import Message


class SpecialEnvironmentGetterCodeTrainData(TypedDict):
    # TODO: ADD RUN IDENTIFIER

    prompt: List[Message]
    raw_response: str

    code: NotRequired[str]

    current_attempt: int
    max_attempts: int

    # Result from `genner.extract_code`
    extract_result: NotRequired[str]

    # Result from `validate_code_offline`, None or error message if there's an error
    validation_result: NotRequired[str]

    # Result from `run_code_in_con`, None if there's no problem
    in_container_run_type: NotRequired[Literal["ok", "err"]]
    in_container_run_result: NotRequired[str]


class StrategyListTrainData(TypedDict):
    # TODO: ADD RUN IDENTIFIER

    prompt: List[Message]
    raw_response: str

    strategy_list: NotRequired[List[str]]

    current_attempt: int
    max_attempts: int

    # Result from `genner.extract_code`
    extract_result: NotRequired[str]
