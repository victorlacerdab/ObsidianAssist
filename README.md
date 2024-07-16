# ObsidianAssist

ObsidianAssist is a fully local interface that allows you to interact, query and talk to the data in your privately owned Obsidian Vault.

The implementation relies on `hf transformers` and `sentence transformers` for model loading, inference, embedding and retrieval capabilities.

## Starting the assistant

In order to load your assistant and start using it, either use the provided `config.json` file directly or edit it in order to customize the assistant to your needs.
In the file you can choose which model to load for inference, which model to use for embedding your Vault, and the specific prompt formatting for your use-case.

Currently, the specific prompt formatting suggested for using instruction-tuned models in the Llama3 family is loaded automatically if one of these models is loaded.
Otherwise, it will start a header prompt using a format suggested by [Prompt Engineering Guide](https://www.promptingguide.ai/), which is a great resource for learning more about prompting techniques.
You can easily alter these options by editing the `config.json` file.

## Interfacing with the model

There are four main ways to interact with the Assistant:

+ The `.interact()` method simply asks a question directly to the loaded language model;
+ The `.context_answer()` method allows you to pass a list with filenames from your Vault and use their contents as context to the model; 
+ The `.rag_answer()` method allows you to perform RAG within your Vault;
+ The `.good_morning()` method adheres to my specific workflow in Obsidian. It expects a certain formatting, and can be used to remind you of todo lists and other ideas you have written down in specific, fixed files.

Every time you call one of the methods above, the model remembers the answer and stores it in the `.chat_history` attribute as a dictionary.
Saving and loading histories is not supported. You will lose your chat history when you close the program.

There are two ways to interact with the chat history:

+ The `.oblivion()` method fully erases your current chat history;
+ The `.partial_oblivion()` method allows you to pass the number of steps that you want to delete. Defaults to one if no integer is provided.

## Usage

You can run the program as a command-line assistant.
If you use an interface such as VSCode to run a jupyter notebook and interact with ObsidianAssist you get proper Markdown formatting display for a better visual experience.
