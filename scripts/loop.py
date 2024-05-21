import os
from pathlib import Path

import docker
from dotenv import load_dotenv
from openai import OpenAI

from src.container import safe_detect_env, safe_filelist, safe_folderlist_bfs
from src.prompt import (
    gen_skill_code_openai,
    gen_skill_sets_openai,
    prepare_task_codegen_prompt_openai,
    prepare_taskgen_prompt_openai,
    prepare_env_codegen_prompts_openai,
)
from src.agent import run_loop

if __name__ == "__main__":
    load_dotenv()

    container_id = "gen2-learn"

    docker_client = docker.from_env()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out_con_path = Path("./data/controller/")
    in_con_path = Path("/")
    in_con_folders = safe_folderlist_bfs(docker_client, container_id, in_con_path)
    smol_in_con_folders = in_con_folders[:25] + ["..."] + in_con_folders[-1:]

    basic_env_info = safe_detect_env(docker_client, container_id)

    env_info_prompts = prepare_env_code_prompts_openai(
        in_con_path=in_con_path,
        in_con_files=smol_in_con_folders,
    )

    taskgen_prompts = prepare_taskgen_prompt_openai(
        env_info=basic_env_info,
        in_con_path=in_con_path,
        in_con_files=smol_in_con_folders,
    )

    task_list = gen_skill_sets_openai(openai_client, taskgen_prompts)

    for task in task_list:
        initial_prompts = prepare_task_codegen_prompt_openai(
            task,
            env_info=basic_env_info,
            in_con_path=in_con_path,
            in_con_files=smol_in_con_folders,
        )
        initial_code = gen_skill_code_openai(openai_client, initial_prompts)

        run_loop(
            docker_client=docker_client,
            openai_client=openai_client,
            container_id=container_id,
            task_description=task,
            initial_code=initial_code,
            initial_prompts=initial_prompts,
            skill_folder=out_con_path,
        )
