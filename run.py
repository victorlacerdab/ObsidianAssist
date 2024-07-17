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

VAULT_PATH = '/Home/YourVault'
RAGDB_PATH = '/Home/RAGYourVaultDB'

model_list = {
    0: 'meta-llama/Meta-Llama-3-8B-Instruct',
    1: 'google/gemma-2-27b-it',
    2: 'EleutherAI/gpt-neo-1.3B'
}

MODEL_NAME = model_list[0]
RAG_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
HEADER_PROMPT = 'You are a precise and to the point assistant. Never include preambles or questions. Use proper Markdown formatting.'
CHUNK_TOKEN_LIMIT = 256

oracle = ObsidianOracle(model_name=MODEL_NAME,
                        device=device,
                        rag_model=RAG_MODEL,
                        header_prompt=HEADER_PROMPT,
                        chunk_token_limit=CHUNK_TOKEN_LIMIT,
                        vault_path=VAULT_PATH,
                        ragdb_path=RAGDB_PATH
                        )

def main():

    print("Welcome to the ObsidianAssist CLI. Type 'help' for available commands.")
    print(f'You have loaded {MODEL_NAME} as your language model.')
    print(f'Make sure to encase the arguments to your commands with square brackets: \'interact [This is a prompt.]\'.')

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
            print("- re-embed vault")
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
                args = int(args.replace('[', '').replace(']', ''))
                oracle.partial_oblivion(args)
            else:
                print("Usage: partial_oblivion [num_previous_days]")
        
        elif cmd == 'embedding_oblivion':
            oracle.embedding_oblivion()

        elif cmd == 're-embed_vault':
            answer = input('This will totally erase your Vault\'s embeddings from your computer and initiate the embedding routine. Proceed? (y/n)')
            if answer == 'y':
                oracle.embedding_oblivion()
                oracle.embed_vault_init()
            elif answer == 'n':
                print('You may proceed as before.')

            elif answer not in ['y', 'n']:
                print('Unknown answer. The embeddings have been preserved. Try again.')
        
        elif cmd == 'see_files':
            print(list(oracle.file_dict.keys()))
        
        elif cmd == 'exit':
            break
        
        else:
            print("Invalid command. Type 'help' for available commands.")

if __name__ == "__main__":
    main()