print(r'''
   ___  _           _      _  _                  _              _       _   
  /___\| |__   ___ (_)  __| |(_)  __ _  _ __    /_\   ___  ___ (_) ___ | |_ 
 //  //| '_ \ / __|| | / _` || | / _` || '_ \  //_\\ / __|/ __|| |/ __|| __|
/ \_// | |_) |\__ \| || (_| || || (_| || | | |/  _  \\__ \\__ \| |\__ \| |_ 
\___/  |_.__/ |___/|_| \__,_||_| \__,_||_| |_|\_/ \_/|___/|___/|_||___/ \__|                                                                        
    ''')

import torch
import re
from sentence_transformers import SentenceTransformer
from oracles import ObsidianOracle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

VAULT_PATH = '/Home/siv33/vbo084/VictorVault'
RAGDB_NAME = '/Home/siv33/vbo084/ObsidianAssist/RAGVictorVaultDB'

model_list = {
    0: 'meta-llama/Meta-Llama-3-8B-Instruct',
    1: 'google/gemma-2-27b-it',
    2: 'EleutherAI/gpt-neo-1.3B'
}

model_name = model_list[0]
rag_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
header_prompt = 'You are a precise and to the point assistant. Never include preambles or questions. Use proper Markdown formatting.'

oracle = ObsidianOracle(model_name, device, rag_model, header_prompt, VAULT_PATH, RAGDB_NAME)

def main():

    print("Welcome to the ObsidianAssist CLI. Type 'help' for available commands.")
    print(f'You have loaded {model_name} as your language model.')

    while True:
        command = input("> ").strip()
        
        if not command:
            continue
        
        parts = command.split(' ')
        cmd = parts[0]
        args = ' '.join(parts[1:])
        
        if cmd == 'help':
            print("Available commands:")
            print("- interact [prompt]")
            print("- context_answer [filename1.md; filename2.md; filenamen.md]")
            print("- rag_answer [prompt] [top_k]")
            print("- good_morning [mode] [num_previous_days] The modes can be either 'work' or 'life'.")
            print("- chat_oblivion")
            print("- embedding_oblivion")
            print("- see_files")
            print("- exit")
        
        elif cmd == 'interact':
            if args:
                oracle.interact(" ".join(args))
            else:
                print("Usage: interact [prompt]")
        
        elif cmd == 'context_answer':
            if args:
                cleaned_args = args.split(']')
                prompt = cleaned_args[0][1:]
                fnames = [fname.lstrip() for fname in cleaned_args[1][2:].split(';')]

                oracle.context_answer(prompt, fnames)
            else:
                print("Usage: context_answer [prompt] [filename1.md; filename2.md; filenamen.md]")
        
        elif cmd == 'rag_answer':
            try:
                cleaned_args = args.replace('[', '').replace(']', '')
                top_k = int(cleaned_args.split(' ')[-1])
                prompt = ' '.join(cleaned_args.split(' ')[:-1])
                oracle.rag_answer(prompt, top_k)
            except:
                print("Usage: rag_answer [prompt] [top_k]")
                pass
        
        elif cmd == 'good_morning':
            if args:
                cleaned_args = args.replace('[', '').replace(']', '')
                args = cleaned_args.split()
                mode = args[0]
                num_previous_days = int(args[1])
                oracle.good_morning(mode, num_previous_days)
            else:
                print("Usage: good_morning [mode] [num_previous_days]")
        
        elif cmd == 'chat_oblivion':
            oracle.oblivion()

        elif cmd == 'partial_oblivion':
            if args:
                oracle.partial_oblivion(args)
            else:
                print("Usage: partial_oblivion [num_previous_days]")
        
        elif cmd == 'embedding_oblivion':
            oracle.embedding_oblivion()
        
        elif cmd == 'see_files':
            print(list(oracle.file_dict.keys()))
        
        elif cmd == 'exit':
            break
        
        else:
            print("Invalid command. Type 'help' for available commands.")

if __name__ == "__main__":
    main()