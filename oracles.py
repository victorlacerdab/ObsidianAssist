import transformers
import torch
import os
import regex as re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from IPython.display import display, Markdown

class MainOracle():
    def __init__(self, model_name, device, header_prompt):
        self.model_name = model_name
        self.pipeline = self.create_pipeline(model_name, device)
        self.chat_history = {}
        self.header_prompt = self.header_prompt_formatting(header_prompt)
        self.running_prompt = None
        self.num_turns = 0
    
    def create_pipeline(self, model_name, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        pipe = pipeline(model=model, task="text-generation", tokenizer=tokenizer, device=device)
        return pipe
    
    def interact(self, new_prompt, max_tokens=250):
        if self.num_turns == 0:
            new_prompt = self.header_prompt + ' ' + self.prompt_formatting(new_prompt, 'user')
            model_answer = self.pipeline(new_prompt, max_new_tokens=max_tokens)[0]['generated_text'][len(new_prompt):]
            model_answer = self.prompt_formatting(model_answer, 'model')
            self.chat_history[self.num_turns] = {new_prompt:model_answer}
            self.num_turns += 1
            try:
                self.display_text(model_answer)
            except:
                print(model_answer)

            self.running_prompt = new_prompt + ' ' + model_answer
 
        else:
            new_prompt = self.prompt_formatting(new_prompt, 'user')
            full_prompt = self.running_prompt + ' ' + new_prompt
            model_answer = self.pipeline(full_prompt, max_new_tokens=max_tokens)[0]['generated_text'][len(full_prompt):]
            model_answer = self.prompt_formatting(model_answer, 'model')
            self.chat_history[self.num_turns] = {new_prompt: model_answer}
            self.num_turns += 1
            try:
                self.display_text(model_answer)
            except:
                print(model_answer)
            
            self.running_prompt = self.running_prompt + ' ' + new_prompt + model_answer

    def oblivion(self):
        self.running_prompt = None
        self.chat_history = {}
        self.num_turns = 0
        print('Oblivion has come to the oracle.')

    def partial_oblivion(self, steps_to_unroll: int):
        
        self.running_prompt = ''
        keys_to_delete = [k for k,_ in self.chat_history.items() if k >= self.num_turns - steps_to_unroll]

        for key in keys_to_delete:
            del self.chat_history[key]
        
        for step in self.chat_history.values():
            for user_msg, model_msg in step.items():
                self.running_prompt = self.running_prompt + user_msg + ' ' + model_msg + ' '

        self.num_turns = max([k for k in self.chat_history.keys()]) + 1

        print(f'Sent the last {steps_to_unroll} to oblivion.')

    def prompt_formatting(self, prompt: str, actor: str) -> str:
        if actor not in {"user", "model", "system"}:
            raise ValueError("Actor arg must be 'user', 'model' or 'system'.")
        
        if self.model_name == 'meta-llama/Meta-Llama-3-8B' or self.model_name == 'meta-llama/Meta-Llama-3-8B-Instruct':
            if actor == 'user':
                prompt = prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
            elif actor == 'model':
                prompt = prompt + '<|eot_id|><|start_header_id|>user<|end_header_id|>'
            elif actor == 'system':
                start_header_str = '<|begin_of_text|><|start_header_id|>system<|end_header_id|> ' 
                end_header_str = ' <|eot_id|><|start_header_id|>user<|end_header_id|>'
                prompt = start_header_str + prompt + end_header_str

        else:
            prompt = '<start_of_turn>user ' + prompt + ' <end_of_turn> ' + '<start_of_turn>model' 
        return prompt
    
    def header_prompt_formatting(self, header_prompt: str) -> str:

        if self.model_name == 'meta-llama/Meta-Llama-3-8B' or 'meta-llama/Meta-Llama-3-8B-Instruct':
            start_header_str = '<|begin_of_text|><|start_header_id|>system<|end_header_id|> ' 
            end_header_str = ' <|eot_id|><|start_header_id|>user<|end_header_id|>'
            return start_header_str + header_prompt + end_header_str
        
        else:
            return header_prompt
    
    def display_text(self, text):
        if self.model_name == 'meta-llama/Meta-Llama-3-8B' or self.model_name == 'meta-llama/Meta-Llama-3-8B-Instruct':
            text2display = text.split('<|eot_id|>')[0].strip()
            display(Markdown(f'**Oracle Response:** {text2display}'))
        else:
            text2display = text.split('<end_of_turn>')[0].strip()
            display(Markdown(f'**Oracle Response:** {text2display[len('<start_of_turn>user'):]}'))

class ObsidianOracle(MainOracle):
    def __init__(self, model_name: str, device: torch.device,
                 header_prompt: str, vault_path: str,
                 ragdb_foldername: str):
        super().__init__(model_name, device, header_prompt)
        self.vault_path = vault_path
        self.ragdb_path = ragdb_foldername
        self.file_dict = self.get_file_dict()
        self.dir_dict = self.get_dir_dict()

    def rag_answer(self, prompt: str) -> str:
        pass

    def context_answer(self, prompt: str, fnames: list) -> str:
        
        contextual_prompt = None
        for fname in fnames:
            pass

        pass

    def edit_file(self, prompt: str, fname: str):
        fpath = self.file_dict[fname]

        with open(fpath, 'r') as f:
            file = f.read()

        pass

    def get_file_dict(self) -> dict:
        file_dict = {}
        for root, _, files in os.walk(self.vault_path, topdown=True):
            for name in files:
                if not name.startswith('.') and name.endswith('.md'):
                    file_path = os.path.join(root, name)
                    file_dict.update({name: file_path})
        
        return file_dict

    def get_dir_dict(self) -> dict:
        dir_dict = {}
        for root, dirs, _ in os.walk(self.vault_path, topdown=True):
            for name in dirs:
                if not name.startswith('.'):
                    file_path = os.path.join(root, name)
                    dir_dict.update({name: file_path})
        
        return dir_dict
    
    ''' 
    Right now the program expects the files to be name 'Life Todo.md' and 'Work Todo.md', and to be placed in the root folder.
    Maybe extend it to be more custom in the future.
    '''

    def good_morning(self, mode: str, num_previous_days: int):
        
        if mode not in ('life', 'work'):
            raise ValueError("Invalid mode. Expected 'life' or 'work'.")

        life_morning_dir = os.path.join(self.vault_path, 'Life Todo.md')
        work_morning_dir = os.path.join(self.vault_path, 'Work Todo.md')

        good_morning_prompt = ' \'[ ]\' means unfinished task while \'[x]\' means an already completed task, do not remind those. Break answer into *Open tasks:* and *Suggestions:*. Dont just repeat what is written. Based your answer on the following: '
        good_morning_prompt = self.prompt_formatting(good_morning_prompt, 'system')

        if mode == 'life':
            with open(life_morning_dir, 'r') as f:
                file = f.read()

            content = self.extract_gmorning_content(file, num_previous_days)
            self.interact(good_morning_prompt + content, max_tokens=500)

        elif mode == 'work':
            with open(work_morning_dir, 'r') as f:
                file = f.read()
            
            content = self.extract_gmorning_content(file, num_previous_days)
            self.interact(good_morning_prompt + content, max_tokens=500)
    
    def extract_gmorning_content(self, file_text, num_previous_days: int) -> str:

        day_sections = re.split(r'^# \d{2}/\d{2}/\d{2}', file_text, flags=re.MULTILINE)
        day_sections = [section.strip() for section in day_sections if section.strip()]

        date_headers = re.findall(r'^# \d{2}/\d{2}/\d{2}', file_text, flags=re.MULTILINE)
        full_sections = [f"{date_headers[i]}\n{day_sections[i]}" for i in range(len(day_sections))]

        relevant_sections = full_sections[-num_previous_days:]
        content = "\n\n".join(relevant_sections)

        return content



