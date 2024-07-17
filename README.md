# ObsidianAssist

```
   ___  _           _      _  _                  _              _       _   
  /___\| |__   ___ (_)  __| |(_)  __ _  _ __    /_\   ___  ___ (_) ___ | |_ 
 //  //| '_ \ / __|| | / _` || | / _` || '_ \  //_\\ / __|/ __|| |/ __|| __|
/ \_// | |_) |\__ \| || (_| || || (_| || | | |/  _  \\__ \\__ \| |\__ \| |_ 
\___/  |_.__/ |___/|_| \__,_||_| \__,_||_| |_|\_/ \_/|___/|___/|_||___/ \__|
``` 


ObsidianAssist is a fully local interface that allows you to interact, query and talk to the data in your privately owned Obsidian Vault.

The implementation relies on `hf transformers` and `sentence transformers` for model loading, inference, embedding and retrieval capabilities.

## Setting-up the assistant

First, install the requirements.txt file in a conda environment.

If you wish to execute the assistant as a CLI, download all the files provided in repository and open the file `run.py`. Remember to activate the proper conda environment before doing so.

There you need to provide your Vault's path to the `VAULT_PATH` variable, and provide a path to `RAGDB_PATH`, where the system will store the vector database as a numpy array and two .json files to handle RAG capabilities.

In the `run.py` file you can also choose any language model from the HF Hub that has been trained to perform `"text-generation"` tasks by changing the `MODEL_NAME` variable.
For embedding your data, you can use any model from the `sentence transformers` library and pass the desired token limit for chunking.

Currently, the specific prompt formatting suggested for using instruction-tuned models in the Llama3 family is loaded automatically if one of these models is loaded.
Otherwise, it will start a header prompt using a format suggested by [Prompt Engineering Guide](https://www.promptingguide.ai/), which is a great resource for learning more about prompting techniques.

Alternatively, you can also interact with the assistant inside a jupyter notebook by initializing a `ObsidianOracle` object and calling its methods directly.
When using this method the oracle's answers will be displayed in neat Markdown format rendering.
The notebook `oracle.ipynb` provides some examples of usage.

## Interfacing with the model


### Prompting the Assistant

+ The `.interact()` method simply asks a question directly to the loaded language model;
+ The `.context_answer()` method allows you to pass a list with filenames from your Vault and use their contents as context to the model; 
+ The `.rag_answer()` method allows you to perform RAG within your Vault;
+ The `.good_morning()` method adheres to my specific workflow in Obsidian. It expects a certain file structure and formatting.

Every time you call one of the methods above, the model remembers the answer and stores it in the `.chat_history` attribute as a dictionary.
Saving and loading histories is not supported. You will lose your chat history when you close the program.

### Interacting with chat history

+ The `.oblivion()` method fully erases your current chat history;
+ The `.partial_oblivion()` method allows you to pass the number of steps that you want to delete. Defaults to one if no integer is provided.

### Managing your embeddings

+ The `.embedding_oblivion()` method fully deletes all files related to your data's embeddings;
+ The `.restart_embedding()` method deletes your files and initiates a new embedding procedure.

## Limitations

The program does not currently track edits made inside your files.
This means that if you add new text it will not be present in your vector database for RAG, and if you delete some text it may still be present in your RAG database.
The program does track when new files are added, but embedding them means re-embedding all of your files.

## Roadmap

I am currently working on making the embedding process more streamlined and efficient by allowing the user to embed or delete specific files.
I am also looking into expanding ObsidianAssist's RAG capabilities to merge current best practices, since right now the program only performs naive chunking and retrieval.



