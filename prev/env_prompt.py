from langchain.prompts import PromptTemplate

ENV_PROMPT = """Given the environment:
current directory is {current_dir} 
available memory of current directory is {current_avail} MB
storage for code is {storage_code} MB

"""

ENV_PROMPT_TEMPLATE = PromptTemplate(
    template=ENV_PROMPT,
    input_variables=["current_dir", "current_avail", "storage_code"],
)
