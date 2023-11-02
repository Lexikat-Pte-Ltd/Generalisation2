import pprint
import pickle
import os
import psutil
import openai
import re
import subprocess
import shutil
import traceback
import jsonlines
import openai
import time
import logging
import json
from subprocess import TimeoutExpired
from env_prompt import ENV_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO,
                    filename="/logs/app.log",
                    filemode='a',
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

class Gen_Agent:
    def __init__(self,skill_storage_pth,train_space = None):
        # record all learned skills, {key:task, value:code}
        self.skills_recorder = {}
        # record of the reward for each skill {key:task, value:reward}
        self.rewards = {}
        # flag is True when there is enough memory; False when there is less than 0.5 MB total memory
        self.mem_flag = True
        self.ori_train_space =  train_space or os.getcwd()
        self.epoch_space = self.ori_train_space
        self.epoch_count = 0
        self.train_space = self.epoch_space
        self.skill_space = skill_storage_pth
        self.skills_to_learn = None      
        self.model = 'text-davinci-003'
        self.env_prompt = ""
        # environment information
        self.environment_info = {}
        self.detect_env()
        
    #@staticmethod   
    def detect_env(self):
        # Calculate memory used by memory directory
        mem_dir_size = self.get_dir_size(self.skill_space)
        # Calculate memory used by train directory
        explore_dir_size = self.get_dir_size(self.epoch_space)
        # Calculate total and available memory of computer
        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        #set space for explore
        partition = available_memory//2
        self.environment_info['memory_space']= mem_dir_size
        self.environment_info['explore_space']= available_memory - partition
        self.environment_info['total_memory']= total_memory
        self.environment_info['available_memory']= available_memory
        self.env_prompt = ENV_PROMPT_TEMPLATE.format(current_dir=self.train_space,
                                                    current_avail=self.environment_info['explore_space'],
                                                    storage_code=self.environment_info['available_memory'])
        pprint.pprint(self.environment_info, width=1)

    def update_env(self):
        # for script in self.skills_recorder.values():
        #     full_pt = self.epoch_space+'/temp_script.py'
        #     try:
        #         with open(full_pt, 'w') as f:
        #             f.write(script)
        #     except FileNotFoundError:
        #         os.makedirs(self.train_space)
        #         with open(full_pt, 'w') as f:
        #             f.write(script)
        #     # run the script
        #     os.system('python '+ full_pt)
        #     # Remove the temporary script file
        #     os.remove(full_pt)            
        self.epoch_count +=1
        print(f'epoch {self.epoch_count} has finished')
        self.detect_env()
        shutil.rmtree(self.epoch_space,ignore_errors=True)

    def backup(self):
        # Define the name of the temp directory for execute single task
        new_space = "temp"
        # Specify the full path of the new directory
        self.train_space = os.path.join(self.epoch_space, new_space)
        # Create a copy of the current directory
        shutil.copytree(self.epoch_space, self.train_space)

    def backup_epoch(self):
        # Define the name of the temp directory for execute single task
        new_space = "epoch"
        # Specify the full path of the new directory
        self.epoch_space = os.path.join(self.ori_train_space, new_space)
        # Create a copy of the current directory
        shutil.copytree(self.ori_train_space, self.epoch_space)

    # This function is to calculate size of directory in MB
    @staticmethod    
    def get_dir_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try :
                    total_size += os.path.getsize(fp)
                except Exception as err:
                    print('An exception happend : ' + str(err))
        # Convert bytes to MB
        return total_size / (1024 * 1024) 

    # This function generate new available skills
    def generate_new_skills(self):
        # Update Skills Recorder if available memory space is more than 20 MB
        if self.environment_info['available_memory'] >= 20 and self.environment_info['explore_space'] >= 18:
            new_skill_sets = self.generate_task_list()
            self.skills_to_learn = new_skill_sets
        else:
            self.mem_flag = False
    
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
    
    # This function is to interact with API to get available actions set
    def generate_task_list(self):
        env_prompt = f"current directory is {self.train_space} and available memory of current directory: {self.environment_info['explore_space']} MB."\
            f"storage for code is {self.environment_info['available_memory']} MB, the task code size shall not exceed storage limit "
        
        prompt = "generate a few speific potential tasks description an AI program can achieve to free up some space based on below limits:"
        input = prompt + env_prompt
        response = openai.Completion.create(
            model=self.model,
            prompt = input,
            max_tokens=500,
            temperature=0.5
            )
        
        new_skill_sets = response['choices'][0]['text'].replace("\n","")
        new_skill_sets = re.split(r'(?!^)(?=(?<!\d)\d+.)', new_skill_sets)
        new_skill_sets = [re.sub(r'\d+\.', '', string) for string in new_skill_sets]
        new_skill_sets
        logging.info(f"generate_task_list {new_skill_sets}")
        return new_skill_sets

    # This function is to interact with API to get code to implement tasks
    def generate_task_code(self,new_skill):
        self.detect_env()
        prompt = "Generate python code for the task (print out code only). Do not generate any comments :\n"
        task_prompt = self.env_prompt + new_skill
        input = prompt + task_prompt
        response = openai.Completion.create(
            model=self.model,
            prompt = input,
            max_tokens=500,
            temperature=0.5
            )
        code = response['choices'][0]['text']
        logging.info(f"code {code}")
        return code

    # generate traning data files in jasonl and return count of traning files
    @staticmethod
    def generate_jsonl_files(task_code_pairs, output_file, pairs_per_file):
        total_pairs = len(task_code_pairs)
        file_count = (total_pairs + pairs_per_file - 1) // pairs_per_file

        for file_index in range(file_count):
            start_index = file_index * pairs_per_file
            end_index = min((file_index + 1) * pairs_per_file, total_pairs)
            filename = f"{output_file}_{file_index}.jsonl"

            with jsonlines.open(filename, mode='w') as writer:
                for index in range(start_index, end_index):
                    task,code = task_code_pairs[index]
                    data = {
                        'prompt': task,
                        'completion': code
                    }
                    writer.write(data)
        return file_count

    #######################################
    ### test, correct & evaluate code   ###
    #######################################

    # reward for available action sets for training
    def get_reward(self,script):
        full_pt = self.train_space+'/temp_script.py'
        # write the script to a temporary file
        try:
            with open(full_pt, 'w') as f:
                f.write(script)
        except FileNotFoundError:
            os.makedirs(self.train_space)
            with open(full_pt, 'w') as f:
                f.write(script)

        # measure the free space before running the script
        free_space_before = os.statvfs('/')[0] * os.statvfs('/')[4]

        # run the script
        os.system('python '+full_pt)

        # measure the free space after running the script
        free_space_after = os.statvfs('/')[0] * os.statvfs('/')[4]

        # calculate the amount of space freed up by the script
        space_freed_up = free_space_after - free_space_before

        # Remove the temporary script file
        os.remove(full_pt)

        return space_freed_up


    # def correct_err(self,script):
    #     proc = subprocess.Popen(['python', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = proc.communicate()
    #     out = out.decode('utf-8')
    #     if err:
    #         err = err.decode('utf-8')
    #     else:
    #         err = False # the code is error free
    #         return err
        
    #     prompt = f"correct python script in string format: {script}, the error for the script is {err} and output is {out}. Print out code only\n"
    #     input = prompt 
    #     response = openai.Completion.create(
    #         model=self.model,
    #         prompt = input,
    #         max_tokens=500,
    #         temperature=0
    #         )
    #     c_script = response['choices'][0]['text']
    #     logging.info(f"correct_err {c_script}")
    #     return c_script


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
            prompt = f"correct python script in string format: {script}, the error for the script is {err} on line {script_per_line[err_line-1]} and output is {out}. Print out code only\n"
            input = self.env_prompt + prompt 
            response = openai.Completion.create(
                model=self.model,
                prompt = input,
                max_tokens=500,
                temperature=0
                )
            script = response['choices'][0]['text']
            if script == old_script:
                return 0,  False
            old_script = script
            logging.info(f"correct_err {script}")
            count  +=1
    
    def do_better(self,script,new_skill):
        self.detect_env()
        prompt = f"Generate another python code that could be combined with this code {script} for the task {new_skill}. Do not generate any comments :\n"
        task_prompt = self.env_prompt + prompt
        # input = prompt + task_prompt
        response = openai.Completion.create(
            model=self.model,
            prompt = task_prompt,
            max_tokens=500,
            temperature=0.5
            )
        code = response['choices'][0]['text']
        logging.info(f"do better {code}")
        return code

    ###################################################
    ### methods for Generalized Agent to self study ###
    ##################################################

    def learn(self):
        if not self.skills_to_learn:
            raise Exception('There is no new skill to learn!')
        for new_task in self.skills_to_learn:
            self.backup()
            script = self.generate_task_code(new_task)
            # AI writes script and tests it.
            freed_space, test_result_success  = self.test(script,new_task)
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
    
    # combine exisitng skills to complete complex task
    # def apply(self, new):
    #     for ext_task in self.skills_recorder.keys():
    #         cmb_task = self.combine(ext_task,new)
    #         script = self.generate_task_code(cmb_task)
    #         test_result_script, test_result_success = self.test(script,cmb_task)
    #         # count=0
    #         # # try 10 times before giving up
    #         # num=10
    #         # while test_result != True and count<=num:
    #         #     if self.correct_err(script):
    #         #         script = self.correct_err(script)
    #         #     test_result = self.test(script,cmb_task)
    #         #     count+=1
    #         if test_result_success == True:
    #             self.skills_recorder[cmb_task] = script
    #             self.rewards[cmb_task] = self.get_reward(script)
                


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

    




###################################

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
