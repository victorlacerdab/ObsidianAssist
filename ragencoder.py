import torch
import numpy as np
import os
import regex as re
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import semantic_search

class ObsidianRAG():
    def __init__(self, rag_model: SentenceTransformer, token_limit: int, file_dict: dict):
        self.model = rag_model
        self.token_limit = token_limit
        self.file_dict = file_dict
        self.embedding_to_file_dict = {}

    def embed_vault(self):

        ''' 
        The file_dict object contains (fname: full_path) pairs.
        '''

        vector_database = []
        emb_chunk_dict = {}

        for fname, file_directory in self.file_dict.items():
                with open(file_directory, 'r') as f:
                    content = f.read()
                    chunked_content = self.chunk(content, self.token_limit) # This returns a list containing strings
                    embeddings = self.embed_from_chunked(chunked_content)
                    emb_chunk_dict.update({str(embeddings[i][:10]):chunked_content[i] for i in range(len(chunked_content))})
                    vector_database.extend(embeddings)

        return np.array(vector_database), emb_chunk_dict
                    
    def embed_from_chunked(self, chunks: list):
        embeddings = [self.model.encode(chunk) for chunk in chunks]
        return embeddings

    def embed_single_file(self, filename: str) -> np.array:

        # First time embedding file

        # Making embeding changes to file
        
        pass

    def chunk(self, content: str, token_limit: int) -> list:

        '''
        The 'sentence-transformers/all-MiniLM-L6-v2' was trained on 128 tokens, which we use as a hard limiter.
        If we simply call the model.embed() method on a chunk of len(tokenized_chunk) > 128 it will implicitly
        chunk the text in the appropriate format. Nevertheless, I do it explicitly so one can use different models
        later which chunk on a different token limit.

        The function splits the content of Obsidian files in sentences on a period, but chunks them further
        in case they exceed the desired token limit.

        The step for chunking sentences down can be improved, since right now it may chunk things in a meaningless way.
        '''
        
        chunked_content = re.split(r'[.;+]', content)
        chunked_content = [sentence.strip() for sentence in chunked_content if sentence.strip()]
        len_chunks = [self.model.tokenize(chunk)['input_ids'].shape[0] for chunk in chunked_content]
        final_chunked_content = []

        for i, chunk in enumerate(chunked_content):
            if len_chunks[i] <= token_limit:
                final_chunked_content.append(chunk)
            else: # Improve this later
                chunk_ratio = len_chunks[i] // token_limit
                for i in range(chunk_ratio):
                    temp_chunk = chunk[i * token_limit:i+1 * token_limit]
                    final_chunked_content.append(temp_chunk)
                
        return final_chunked_content
    
    def similarity_scores(self, embedded_query: torch.tensor, vector_db: torch.tensor, top_k: int) -> list[dict]:
        scores = semantic_search(query_embeddings=embedded_query, corpus_embeddings=vector_db, top_k=top_k)
        return scores

