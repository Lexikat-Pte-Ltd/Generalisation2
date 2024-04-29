import logging
import os
import psutil
import shutil
import pickle
import pprint

import openai
from langchain.llms import OpenAI 
from langchain.chains import LLMChain

import re

from prev.env_prompt import ENV_PROMPT_TEMPLATE

tool_kit = {"google_cloud":{"api_key":""}} # there are also other tools we can use to store the files

logging.basicConfig(level=logging.INFO,
                    filename="/logs/app.log",
                    filemode='a',
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

class Gen_Agent:
    def __init__(self, train_path = "/home", external_path="/agent_external"): 
        self.train_path = train_path
        self.external_path = external_path
        self.logs_path = self.make_dir(external_path + "/logs")
        self.learned_skills_path = self.make_dir(external_path + "/learned_skills")
        self.train_backup_path = self.make_dir(external_path + "/train_backup_path")

        self.environment_info = {}
        self.env_prompt = ""

        self.init_logging()
        self.detect_env()
        
    
    # detect_env is to detect the size of the system's memory for storaging the script and running the script
    def detect_env(self):
        total_system_memory = psutil.virtual_memory().total // (1024 * 1024)
        available_system_memory = psutil.virtual_memory().available // (1024 * 1024)
        running_memory = available_system_memory//2
        storage_space = available_system_memory - running_memory

        self.environment_info['total_system_memory'] = total_system_memory
        self.environment_info['available_system_memory'] = available_system_memory
        self.environment_info['running_memory'] = running_memory
        self.environment_info['storage_space'] = storage_space

        # current_dir is the space where the agent try to minimize the disk usage
        # current_avail is the available memory for running the script
        # storage_code is the memory space for scripts' subprocessing
        self.env_prompt = ENV_PROMPT_TEMPLATE.format(current_dir=self.train_path,
                                                     current_avail=running_memory,
                                                     storage_code=storage_space)
        pprint.pprint(self.environment_info, width=1)

    def training_path_backup(self):
        shutil.copytree(self.train_path, "/train_path_backup")

    def backup_epoch(self):
        # Define the name of the temp directory for execute single task
        new_space = "epoch"
        # Specify the full path of the new directory
        self.epoch_space = os.path.join(self.ori_train_space, new_space)
        # Create a copy of the current directory
        shutil.copytree(self.ori_train_space, self.epoch_space)

    def generate_task_list(self):
        all_files = self.get_files_list(self.train_path)
        env_prompt = f'Here is the files distribution in the directory {self.train_path}: \n{all_files}. \n' \
                     f'And the OS environment information is: '\
                     f'available memory for running a script = {self.environment_info["running_memory"]} MB, '\
                     f'storage space left for script code is {self.environment_info["storage_space"]} MB, the task code size shall not exceed storage limit.'\
                     
        prompt = f"generate a few speific potential tasks description an AI program can achieve to free up some disk space for the directory {self.train_path}. Only generate the task:\n"
        input = env_prompt + prompt + "\nTask:"

        from langchain import  PromptTemplate
        template = "{question}"
        prompt = PromptTemplate(template=template, input_variables=['question'])

        llm = OpenAI(openai_api_base="https://api.openai.com/v1",
             openai_api_key='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')
        
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.invoke({"question":input})['text']

        # response = openai.Completion.create(
        #     model=self.model,
        #     prompt = input,
        #     max_tokens=500,
        #     temperature=0.5
        #     )
        logging.info(f"[PROMPT] generate_task_list {input}")
        
        new_skill_sets = response.split("\n")
        new_skill_sets = [re.sub(r'\d+\.', '', string) for string in new_skill_sets]

        logging.info(f"generate_task_list {new_skill_sets}")
        return new_skill_sets
    

    def generate_task_code(self,new_skill):
        if not self.enough_memory():
            return False
        
        prompt = f"Generate python code for the task: {new_skill}\n Print out code only. Do not generate any comments :\nCode:"
        
        input = self.env_prompt + prompt
        response = openai.Completion.create(
            model=self.model,
            prompt = input,
            max_tokens=500,
            temperature=0.5
            )
        logging.info(f"[PROMPT] code {input}")
        code = response['choices'][0]['text']
        code = self.preprocess_code(code)
        logging.info(f"code {code}")
        return code

    # Function to save the skills_recorder to a local file
    def save_skills_recorder(self):
        full_path = os.path.join(self.skill_space, 'skills_recorder.pkl')
        with open(full_path, 'wb') as file:
            pickle.dump(self.rewards, file)

    # Function to load the skills_recorder from memory or a local file
    def load_skills_recorder(self, opt = None):
        full_path = os.path.join(self.skill_space, 'skills_recorder.pkl')
        if opt=='local':
            try:
                with open(full_path, 'rb') as file:
                    data = pickle.load(file)
                return data
            except FileNotFoundError:
                # Return an empty dictionary if the file doesn't exist
                return {}
        else:
            return self.skills_recorder

    def test(self,script,new_task):
        old_script = script
        count=0
        num=7
        
        while True:
            if count  == num :
                return 0, False
            proc = subprocess.Popen(['python', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            free_space_before = os.statvfs('/')[0] * os.statvfs('/')[4]
            
            try:
                out, err = proc.communicate(timeout=90)
                # measure the free space after running the script
                free_space_after = os.statvfs('/')[0] * os.statvfs('/')[4]

                # calculate the amount of space freed up by the script
                space_freed_up = free_space_after - free_space_before
                out = out.decode('utf-8')
            
                if err:
                    err = err.decode('utf-8')
                else:
                    err = False # the code is error free
                    return space_freed_up, True
            except TimeoutExpired:
                proc.kill()
                out = ""
                err = "subprocess.TimeoutExpired"

            err_line = 0
            for line in err.split('\n'):
                if (line.strip().startswith('File "<string>')):
                    stripped = line.strip()
                    found = re.search(r'^File \"\<string\>\"\, line ([0-9]+)',stripped)
                    err_line = int(found.group(1))
            
            script_per_line = script.split('\n')
            logging.info(f"err {err} \nerr_line {err_line} {script_per_line[err_line-1]}")
            self.detect_env()
            prompt = f"correct python script in string format: {script}, the error for the script is {err} on line {script_per_line[err_line-1]} and output is {out}. Print out code only\nCode:"
            input = self.env_prompt + prompt 
            
            response = openai.Completion.create(
                model=self.model,
                prompt = input,
                max_tokens=500,
                temperature=0
                )
            logging.info(f"[PROMPT] correct_err {input}")
            script = response['choices'][0]['text']
            if script == old_script:
                return 0,  False
            old_script = script
            script = self.preprocess_code(script)
            logging.info(f"correct_err {script}")
            count  +=1
    
    def do_better(self,script,new_skill):
        self.detect_env()
        prompt = f"Generate another python code that could be combined with this code {script} for the task {new_skill}. Do not generate any comments :\nCode:"
        task_prompt = self.env_prompt + prompt
        # input = prompt + task_prompt
        response = openai.Completion.create(
            model=self.model,
            prompt = task_prompt,
            max_tokens=500,
            temperature=0.5
            )
        logging.info(f"[PROMPT] do better {task_prompt}")
        code = response['choices'][0]['text']
        code = self.preprocess_code(code)
        logging.info(f"do better {code}")
        return code

    ###################################################
    ### methods for Generalized Agent to self study ###
    ##################################################

    def learn(self):
        if not self.skills_to_learn:
            raise Exception('There is no new skill to learn!')
        for new_task in self.skills_to_learn:
            if len(new_task.strip()) == 0:
                continue
            self.backup()
            script = self.generate_task_code(new_task)
            # AI writes script and tests it.
            freed_space, test_result_success  = self.test(script,new_task)
            logging.info(f'task {new_task} status: {test_result_success}\nfreed Space: {freed_space}')
            # logging.info(f"reward {self.rewards}")
            # count=0
            # try 10 times before giving up
            # num=10
            
            # while test_result != True and count<=num:
            #     if self.correct_err(script):
            #         script = self.correct_err(script)
            #     test_result = self.test(script,new_task)
            #     count+=1

            if test_result_success == True:
                # Script runs but fails to free up space
                if freed_space <= 0:
                    better_script = self.do_better(script,new_task)
                    better_freed_space, test_result_success  = self.test(better_script,new_task)

                    if test_result_success == True:
                        if better_freed_space > 0:
                            self.skills_recorder[new_task] = script
                            self.rewards[new_task] = (script,freed_space)
                else:
                    # self.apply(new_task)
                    self.skills_recorder[new_task] = script
                    self.rewards[new_task] = (script,freed_space)

                    # break              
            shutil.rmtree(self.train_space,ignore_errors=True)
            self.train_space = self.epoch_space

    def combine(self,task_1,task_2):
        prompt = f"combine these 2 tasks to 1:{task_1}, {task_2}. (compute combined task description only)"
        input = prompt
        response = openai.Completion.create(
            model=self.model,
            prompt = input,
            max_tokens=500,
            temperature=0.5
            )
        cmb_skill = response['choices'][0]['text'].replace("\n","")
        logging.info(f"combine {cmb_skill}")
        return cmb_skill

    # helper function =================================================================================================
    def preprocess_code(self, code):
        """sometimes wizardcoder generate code with monospace thingy and explanation, so we need to delete it"""
        """ Example: ```python
            print("Hello world")
            ```
            Explanation:
            1. The script says hello!"""
        """Expected Result: print("Hello world")"""
        cleaned_code = code[code.find('```python')+9:code.rfind('```')]
        if len(cleaned_code) == 0:
            return code
        else:
            return cleaned_code
        
    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def init_logging(self):
        logging.basicConfig(level=logging.INFO,
                    filename=self.logs_path + "/app.log",
                    filemode='a',
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
        
    def get_files_list(self, path):
        ls = ''
        for root, dirs, files in os.walk(path):
            for file in files:
                # ls.append(os.path.join(root, file))
                ls += os.path.join(root, file) + '\n'

        return ls
        
    def enough_memory(self):
        self.detect_env()
        if self.environment_info["running_memory"] >= 20 and self.environment_info["storage_space"] >= 20:
            return True
        else:
            return False
        

if __name__=="__main__":
    trial = Gen_Agent(skill_storage_pth = '/media', train_space = '/home')
    # while memory >0 
    # for demo only
    for i in range(6)  :
        # input the skills storage path (and train_space maybe; the default traning sapce is current directory)
        trial.backup_epoch()
        trial.generate_new_skills()
        trial.learn()
        # save successful task:code pairs to memory space (skills storage path)
        trial.save_skills_recorder()
        # execute in original training space
        trial.update_env()