import torch
import json
import numpy as np
import os
import regex as re
from ragencoder import ObsidianRAG
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from IPython.display import display, Markdown

class MainOracle():
    def __init__(self, model_name, rag_model, device, header_prompt):
        self.model_name = model_name
        self.rag_model = rag_model
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
    
    def interact(self, new_prompt, max_tokens=800):
        if self.num_turns == 0:
            new_prompt = self.header_prompt + ' ' + self.prompt_formatting(new_prompt, 'user')
            model_answer = self.pipeline(new_prompt, max_new_tokens=max_tokens)[0]['generated_text'][len(new_prompt):]
            model_answer = self.prompt_formatting(model_answer, 'model')
            self.chat_history[self.num_turns] = {new_prompt:model_answer}
            self.num_turns += 1
            try:
                self.display_text(model_answer)
                print(model_answer)
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
                print(model_answer)
            except:
                print(model_answer)
            
            self.running_prompt = self.running_prompt + ' ' + new_prompt + model_answer

    def oblivion(self):
        self.running_prompt = None
        self.chat_history = {}
        self.num_turns = 0
        print('Oblivion has come to the oracle.')

    def partial_oblivion(self, steps_to_unroll = 1):
        
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
                 rag_model, header_prompt: str, chunk_token_limit: int,
                 vault_path: str, ragdb_path: str):
        super().__init__(model_name, rag_model, device, header_prompt)
        self.vault_path = vault_path
        self.ragdb_path = ragdb_path
        self.file_dict = self.get_file_dict()
        self.dir_dict = self.get_dir_dict()
        self.rag_model = ObsidianRAG(rag_model, token_limit=chunk_token_limit, file_dict=self.file_dict)
        self.embedding_db_paths = self.embed_vault_init() # Tuple containing (vectordb_path: np.array, emb_chunk_path: json, embedded_files_path: json)

    def rag_answer(self, prompt: str, top_k: int) -> None:

        embedded_prompt = torch.tensor(self.rag_model.model.encode(prompt)).unsqueeze(0)
        vector_db = np.load(self.embedding_db_paths[0])
        with open(self.embedding_db_paths[1], 'r') as json_file:
            emb_chunk_dict = json.load(json_file)
        
        similarity_score = self.rag_model.similarity_scores(embedded_prompt, torch.tensor(vector_db), top_k)
        chunk_idcs = [entry['corpus_id'] for similarity_result in similarity_score for entry in similarity_result]
        embedding_keys = [str(vector_db[idx, :10]) for idx in chunk_idcs]
        retrieved_chunks = [emb_chunk_dict[emb_key] for emb_key in embedding_keys]
        final_prompt = prompt + 'These are some pieces of information that might be relevant expand upon them: ' + ' '.join(retrieved_chunks)
        rag_answer = self.interact(final_prompt)
        
    def context_answer(self, prompt: str, fnames: list) -> None:
        
        contextual_prompt = []

        for fname in fnames:
            with open(self.file_dict[fname], 'r') as f:
                content = f.read()
                contextual_prompt.append(fname[:len(fname)-3] + ' ' + content)
        
        contextual_prompt = ' '.join(contextual_prompt)
        answer = self.interact(contextual_prompt + ' ' + prompt)
    
    '''
    Right now the program expects the files to be name 'Life Todo.md' and 'Work Todo.md', and to be placed in the root folder.
    Extend it to be more custom in the future.
    '''

    def good_morning(self, mode: str, num_previous_days: int) -> None:
        
        if mode not in ('life', 'work'):
            raise ValueError("Invalid mode. Expected 'life' or 'work'.")

        life_morning_dir = os.path.join(self.vault_path, 'Life Todo.md')
        work_morning_dir = os.path.join(self.vault_path, 'Work Todo.md')

        good_morning_prompt = ' \'[ ]\' means unfinished task while \'[x]\' means an already completed task. Break answer into *Open tasks:* and *Suggestions:*. Dont just repeat what is written, include all open tasks. Base your answer on the following: '
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

        ''' 
        The function expects that the contents of both the Work and Life todo Markdown files
        have '# dd/mm/yy' marking the days, and nothing else merits the h1 header # symbol.
        '''

        day_sections = re.split(r'^# \d{2}/\d{2}/\d{2}', file_text, flags=re.MULTILINE)
        day_sections = [section.strip() for section in day_sections if section.strip()]

        date_headers = re.findall(r'^# \d{2}/\d{2}/\d{2}', file_text, flags=re.MULTILINE)
        full_sections = [f"{date_headers[i]}\n{day_sections[i]}" for i in range(len(day_sections))]

        relevant_sections = full_sections[-num_previous_days:]
        content = "\n\n".join(relevant_sections)

        return content
    
    def embed_vault_init(self) -> list[str]:
        vaultdb_path = os.path.join(self.ragdb_path, 'vaultdb.npy')
        emb_chunk_dict_path = os.path.join(self.ragdb_path, 'embchunk.json')
        files_to_embed_path = os.path.join(self.ragdb_path, 'embedded_fnames')

        if not os.path.exists(self.ragdb_path) or not os.path.exists(vaultdb_path):
            response = input('No vector database associated with the selected Vault. Do you wish to embed your vault? This may take a while. (y/n).')

            if response == 'y':
                vault_vector_db, emb_chunk_dict = self.rag_model.embed_vault()
                self.save_rag_db_disk(vault_vector_db, emb_chunk_dict, vaultdb_path, files_to_embed_path, emb_chunk_dict_path)
                return (vaultdb_path, emb_chunk_dict_path, files_to_embed_path)

            elif response == 'n':
                print('Proceeding without a vector database. RAG functionality will not be available.')
                print('You may call the .embed_vault_init() method to embed your files later.')
                pass

            elif response not in ['y', 'n']:
                raise ValueError('Response must be \'y\' or \'n\'')

        else:
            with open(files_to_embed_path, 'r') as fnames_json:
                prev_embedded_files = [key for key in list(dict(json.load(fnames_json)).keys())]
            
            with open(files_to_embed_path, 'r') as fnames_json:
                old_file_dict = dict(json.load(fnames_json))
            
            old_file_dict = prev_embedded_files
            print(f'Prev_embedded_files: {prev_embedded_files}')
            recent_files = [key for key in list(self.file_dict.keys())]
            print(f'Recent files: {recent_files}')
            deleted_files = [fname for fname in prev_embedded_files if fname not in recent_files]
            print(f'Deleted files: {deleted_files}')
            new_files = [fname for fname in recent_files if fname not in prev_embedded_files]
            print(f'New files: {new_files}')

            if deleted_files or new_files:
                response = input('Files were either included or removed from your vault. Would you like to embed the new ones and delete the embeddings for the old ones? This may take a while. (y/n)')

                if response == 'y':
                    self.embedding_db_paths = [vaultdb_path, emb_chunk_dict_path, files_to_embed_path]
                    self.embedding_oblivion()
                    vault_vector_db, emb_chunk_dict = self.rag_model.embed_vault()
                    self.save_rag_db_disk(vault_vector_db, emb_chunk_dict, vaultdb_path, files_to_embed_path, emb_chunk_dict_path)
                    return (vaultdb_path, emb_chunk_dict_path, files_to_embed_path)
                    
                elif response == 'n':
                    print('Proceeding without making changes to the existing embeddings. New information will not be used for RAG, and deleted chunks might be included in answers.')
                    self.file_dict = old_file_dict
                    return (vaultdb_path, emb_chunk_dict_path, files_to_embed_path)

                elif response not in ['y', 'n']:
                    raise ValueError('Response must be \'y\' or \'n\'')
            
            else:
                return (vaultdb_path, emb_chunk_dict_path, files_to_embed_path)

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

    def save_rag_db_disk(self, embedded_vault: np.array, emb_chunk_dict: dict, vaultdb_path: str, files_to_embed_path: str, emb_chunk_dict_path: str) -> None:
        os.makedirs(self.ragdb_path)
        np.save(vaultdb_path, embedded_vault)
        with open(files_to_embed_path, 'w') as json_file:
                json.dump(self.file_dict, json_file, indent=4)
        with open(emb_chunk_dict_path, 'w') as json_file:
                json.dump(emb_chunk_dict, json_file, indent=4)
        print('The vault has been embedded.')

    def restart_embedding(self) -> None:
        self.embedding_oblivion()
        self.embed_vault_init()

    def embedding_oblivion(self) -> None:
        for path in self.embedding_db_paths:
            os.remove(path)
        os.rmdir(self.ragdb_path)