#!/usr/bin/env python3

# The embedded document Q&A retrieval functionality is from: https://github.com/imartinez/privateGPT.git
# The main functionality from file privateGPT.py has been integrated in BabyAGI
# Many thanks to https://github.com/imartinez for the great work!

from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

import os
import time
import logging
from collections import deque
from typing import Dict, List
import importlib
import openai
import chromadb
import tiktoken as tiktoken
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import re

# default opt out of chromadb telemetry.
from chromadb.config import Settings

# Engine configuration
# --------------------
# Model: GPT, LLAMA, HUMAN, etc.
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000))
LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", 0.9))
LLAMA_CONTEXT = int(os.getenv("LLAMA_CONTEXT", 4000))
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "")
LLAMA_CTX_MAX = int(os.getenv("LLAMA_CTX_MAX", 1024))
LLAMA_THREADS_NUM = int(os.getenv("LLAMA_THREADS_NUM", 8))

# Internet search configuration
ENABLE_SEARCH_EXTENSION = os.getenv("ENABLE_SEARCH_EXTENSION", "false").lower() == "true"
WIKI_SEARCH = os.getenv("WIKI_SEARCH", "false").lower() == "true"

# Document embedding vectorstore configuration
ENABLE_EMBEDDINGS_EXTENSION = os.getenv("ENABLE_EMBEDDINGS_EXTENSION", "false").lower() == "true"
EMBEDDINGS_BACKUP = os.getenv("EMBEDDINGS_BACKUP", "false").lower() == "true"
EMBEDDINGS_UPDATE = os.getenv("EMBEDDINGS_UPDATE", "false").lower() == "true"
WIKI_CONTEXT = os.getenv("WIKI_CONTEXT", "false").lower() == "true"

# Output step-by-step report to file
ENABLE_REPORT_EXTENSION = os.getenv("ENABLE_REPORT_EXTENSION", "false").lower() == "true"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not (LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("human")):
    assert OPENAI_API_KEY, "\033[91m\033[1m" + "OPENAI_API_KEY environment variable is missing from .env" + "\033[0m\033[0m"

# OpenAI model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
openai.api_key = OPENAI_API_KEY

# Table config
PERSISTENT_STORAGE = os.getenv("PERSISTENT_STORAGE", "false").lower() == "true"
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "BabyAGI"))
COOPERATIVE_MODE = os.getenv("COOPERATIVE_MODE", "")
JOIN_EXISTING_OBJECTIVE = False

# Goal configuration
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))


# Extensions support begin
# ------------------------
def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
        os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, LLM_MODEL, DOTENV_EXTENSIONS, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE = parse_arguments()

# Human mode extension
# Gives human input to babyagi
if LLM_MODEL.startswith("human"):
    if can_import("extensions.human_mode"):
        from extensions.human_mode import user_input_await

# Load additional environment variables for enabled extensions
# TODO: This might override the following command line arguments as well:
#    OBJECTIVE, INITIAL_TASK, LLM_MODEL, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions, but also provide command line
# arguments to override them


# Output to file for creation of step-by-step report
# TODO: This is a bit of a hack, but it works for now
if ENABLE_REPORT_EXTENSION:
    REPORT_FILE = os.getenv("REPORT_FILE", "report.txt")
    ACTION = os.getenv("ACTION", "")

    # Write to summary file
    def write_report(file: str, text: str, mode: chr):
        try:
            with open(file, mode) as f:
                f.write(text)
        except Exception as e:
            print(f"Error writing to report file: {e}")      

    # Read from summary file
    def read_report(file: str):
        try:
            with open(file, 'r') as f:
                text = f.read()
                return text
        except Exception as e:
            return f"Error reading from report file: {e}"

    # Check text needs to be written to report file
    def check_report(result: str):
        if ("Task List" or "Task list" and "task list") not in result[0:50]:
            try:
                if "```" in result:
                    if result.count("```") > 1:
                        block_counter = int(result.count("```")/2)
                    else:
                        block_counter = result.count("```")
                    for i in range(block_counter):
                        write_report(REPORT_FILE, "\n```" + result.split("```")[i+1], 'a')

                    write_report(REPORT_FILE, "\n", 'a')
                    print(f'\nCode blocks written to file: {block_counter}')

                elif "###" in result:
                    block_counter = result.count("###")
                    for i in range(block_counter):
                        write_report(REPORT_FILE, "\n###" + result.split("###")[i+1], 'a')

                    write_report(REPORT_FILE, "\n", 'a')
                    print(f'\nReport blocks written to file: {block_counter}')

            except Exception as e:
                print(f"Error in check_report: {e}")

    # Check if report file exists
    def check_report_file(file: str):
        try:
            res = False
            with open(file, 'r') as f:
                lines = f.readlines()
                if OBJECTIVE in lines[3]:
                    print(f"Objective is unchanged, use existing report file and append new text...")
                    res = True
            if not res:
                with open(file, 'w') as f:
                    print("Objective for report file has changed, overwrite file...")
                    f.write(f'# In this file BabyAGI stores information, as configured in .env file under ENABLE_REPORT_EXTENSION.\n# The output is derived step-by-step and new information is appended to the file...\n\nOBJECTIVE: {OBJECTIVE}\n---------------------------\n')
            
        except:
            with open(file, 'w') as f:
                print("Report file does not exists, create file...")
                f.write(f'# In this file BabyAGI stores information, as configured in .env file under ENABLE_REPORT_EXTENSION.\n# The output is derived step-by-step and new information is appended to the file...\n\nOBJECTIVE: {OBJECTIVE}\n---------------------------\n')
    
    check_report_file(REPORT_FILE)
        

# Document embedding with Q&A retrieval using langchain (multiple file types supported)
# Load & embedd all supported document in folder 'source_documents' in chromadb vector store with 'ingest.py'
# The complete functionality is from: https://github.com/imartinez/privateGPT.git
if ENABLE_EMBEDDINGS_EXTENSION or EMBEDDINGS_UPDATE or EMBEDDINGS_BACKUP or WIKI_CONTEXT:
    if can_import("extensions.doc_embedding"):
        from langchain.chains import RetrievalQA
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.vectorstores import Chroma
        from langchain.llms import GPT4All, LlamaCpp
        from extensions.doc_embedding import parse_arguments, text_writer, text_loader

        if WIKI_CONTEXT:
            from langchain.utilities import WikipediaAPIWrapper
            wikipedia = WikipediaAPIWrapper()

        DOC_STORE_NAME = os.getenv("DOC_STORE_NAME", "")
        DOC_SOURCE_PATH = os.getenv("DOC_SOURCE_PATH", "")
        SCRAPE_SOURCE_PATH = os.getenv("SCRAPE_SOURCE_PATH", "")

        persist_directory = os.environ.get("DOC_STORE_NAME", "")
        source_path = os.environ.get("DOC_SOURCE_PATH", "")
        embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
        embeddings_temperature = float(os.environ.get("EMBEDDINGS_TEMPERATURE", 0.8))
        model_type = os.environ.get("EMBEDDINGS_MODEL_TYPE", "LlamaCpp")
        model_path = os.environ.get("EMBEDDINGS_MODEL_PATH", "")
        model_n_ctx = int(os.environ.get("EMBEDDINGS_CTX_MAX", 1024))

        # Define the Chroma settings
        CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )

        # Update document vectorstore
        def update_db():
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
            retriever = db.as_retriever()
            return db, retriever
        
        # --------------------------------------------------------
        # Important: Vectorstore must exist before document embedding Q&A retrieval as context can be used
        # 1.) This can be done either by using ingest.py on a folder with documents...
        # ... or by running BabyAGI with EMBEDDINGS_BACKUP=true, EMBEDDINGS_UPDATE=false and ENABLE_EMBEDDINGS_EXTENSION=false for at least one task result
        # 1b.) Then run at least one cycle with EMBEDDINGS_UPDATE=true and ENABLE_EMBEDDINGS_EXTENSION=false (if ingest.py has not been used for setting up a document embedding vectorstore)
        # 2.) Then ENABLE_EMBEDDINGS_EXTENSION can be set to add document embedding Q&A retrieval as context
        # --------------------------------------------------------
        # Parse the command line arguments and setup document vectorstore
        args = parse_arguments()
        db, retriever = update_db()

        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

        # Prepare the LLM
        match model_type:
            case "LlamaCpp":
                llm_doc_embedd = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_threads=LLAMA_THREADS_NUM, callbacks=callbacks, verbose=False, temperature=embeddings_temperature)
            case "GPT4All":
                llm_doc_embedd = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_threads=LLAMA_THREADS_NUM, callbacks=callbacks, verbose=False, temp=embeddings_temperature)
            case _default:
                print(f"Model {model_type} not supported for document embedding!")
                exit;
        
        qa = RetrievalQA.from_chain_type(llm=llm_doc_embedd, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)


# Internet smart search (based on BabyCatAGI)
# - SERPAPI, Google CSE or browser search and fallback strategy for API key limits
# - Works also w/o any API key
# - Search results are summarized by LLM
if ENABLE_SEARCH_EXTENSION:
    if can_import("extensions.smart_search"):
        from extensions.smart_search import web_search_tool

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
        SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
        INITIAL_SEARCH = os.getenv("INITIAL_SEARCH", "false").lower() == "true"

        def initial_search(search_request: str, task: str, num_results: int, num_requests: int):
            if search_request != "":
                print("\033[93m\033[1m" + "\nInitial action:" + "\033[0m\033[0m" + f" Perform smart internet search and get web scrape results")
                prompt = 'Analyze the following objective for different aspects, topics, tasks.\n'
                prompt += f'Objective: {OBJECTIVE}\n'
                prompt += f'Create {num_requests} concise internet search requests, based on the analysis and respond with a numbered list.\n'
                prompt += '\n\nYour response: '
                response = openai_call(prompt, max_tokens=MAX_TOKENS)

                result = ""
                list = response.split("\n")
                link_list = []
                i = int(0)
                for l in list:
                    if list.index(l) >= 1:
                        try:
                            l = l.split(". ")[1]
                            if l != "":
                                search_request = l
                                i += 1
                                print(f'Extracted search request: {search_request}')    
                                if GOOGLE_API_KEY and GOOGLE_CSE_ID:
                                    print("Access smart search with Google CSE...")
                                    print_to_file("\nAccess smart search with Google CSE...\n", 'a')
                                    search_results, scrape_page, scrape_raw, links = web_search_tool(query=search_request, task=task, num_extracts=num_results, mode="google", summary_mode=False)
                                elif SERPAPI_API_KEY:
                                    print("Access smart search with SERPAPI...")
                                    print_to_file("\nAccess smart search with SERPAPI...\n", 'a')
                                    search_results, scrape_page, scrape_raw, links = web_search_tool(query=search_request, task=task, num_extracts=num_results, mode="serpapi", summary_mode=False)
                                else:
                                    print("Access smart search with www.duckduckgo.com...")
                                    print_to_file("\nAccess smart search with www.duckduckgo.com...\n", 'a')
                                    search_results, scrape_page, scrape_raw, links = web_search_tool(query=search_request, task=task, num_extracts=num_results, mode="browser", summary_mode=False)
                                
                                if scrape_raw != "":
                                    if EMBEDDINGS_BACKUP:
                                        input_text = f'\nInternet search request: {search_request}\n\nWeb page scrape content:\n{scrape_raw}\n\nSources: {links}\n'
                                        file_path = SCRAPE_SOURCE_PATH + f'/scraper/initial_scrape_toplist{i}.txt'
                                        text_writer(file_path=file_path, input=input_text, text="initial web scrape")
                                    
                                    if EMBEDDINGS_UPDATE:
                                        print("\033[93m\033[1m" + "\n*****RESULTS EMBEDDING IN DOCUMENT VECTORSTORE*****" + "\033[0m\033[0m")
                                        print_to_file("\n*****RESULTS EMBEDDING IN DOCUMENT VECTORSTORE*****\n", 'a')
                                        input_memory = f'\nInternet search request: {search_request}\n\nWeb page scrape content:\n{scrape_raw}\n'
                                        res = text_loader(DOC_STORE_NAME, input_memory, links)
                                        if res:
                                            db, retriever = update_db()
                        except:
                            break
        
# Wikipedia API extension (as stand-alone or amendment for internet search)
if WIKI_SEARCH:
    from langchain.utilities import WikipediaAPIWrapper
    
    wikipedia = WikipediaAPIWrapper()
# ------------------------
# Extensions support end


# Print configuration
print("\033[95m\033[1m" + "\n*****CONFIGURATION*****" + "\033[0m\033[0m")
print(f"Name  : {INSTANCE_NAME}")
print(f"Mode  : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}\n")
print(f"LLM Model           : {LLM_MODEL}")
print(f"Max. Tokens         : {MAX_TOKENS}\n")
if LLM_MODEL.startswith("llama"):
    print(f"LLAMA Temperature   : {LLAMA_TEMPERATURE}")
    print(f"LLAMA Context Size  : {LLAMA_CONTEXT}")
    print(f"LLAMA CTX MAX       : {LLAMA_CTX_MAX}\n")
print(f"File Report Extension           : {ENABLE_REPORT_EXTENSION}\n")
print(f"Smart Internet Search Extension : {ENABLE_SEARCH_EXTENSION}")
print(f"  - Supplement Wikipedia Search : {WIKI_SEARCH}\n")
print(f"Document Embedding Extension         : {ENABLE_EMBEDDINGS_EXTENSION}")
print(f"  - Wikipedia Search as add. Context : {WIKI_CONTEXT}")
print(f"  - Update DocStore with TaskResults : {EMBEDDINGS_BACKUP}")
print(f"  - Persistent Entity Memory         : {EMBEDDINGS_UPDATE}\n")
client = chromadb.Client(Settings(anonymized_telemetry=False))

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"
assert INITIAL_TASK, "\033[91m\033[1m" + "INITIAL_TASK environment variable is missing from .env" + "\033[0m\033[0m"


# Function definitions
# ------------------------
# Q&A retrieval with embedded document vector store
def qa_retrieval(task: str, context: list, mode: str):
    doc_context = str("")

    if INITIAL_TASK in task:
        task = INITIAL_TASK

    query = f'Give advice for a query, consisting of an ultimate objective and context.'
    query += f'\nObjective: {OBJECTIVE}'
    query += f'\nContext: ' + '\n'.join(context)
    query += f'\n\nYour response: '

    if mode == "wiki":
        answer = wikipedia.run(query)
        print(f"\033[96m\033[1m\n*****WIKIPEDIA CONTEXT*****\033[0m\033[0m\n{answer}")
        print_to_file(f"\n*****WIKIPEDIA CONTEXT*****\n{str(answer)}\n\n", 'a')
    elif mode == "embedding":
        print(f"\033[96m\033[1m\n*****DOCUMENT EMBEDDING CONTEXT*****\033[0m\033[0m")
        res = qa(query)
        answer, docs = res['result'], [][0:int(LLAMA_CTX_MAX)] if args.hide_source else res[source_path]
        print()
        print_to_file(f"\n*****DOCUMENT EMBEDDING CONTEXT*****\n{str(answer)}\n\n", 'a')
    
    if mode == "wiki" and answer.startswith("No good Wikipedia Search Result was found"):
        doc_context = str("")
    elif mode == "embedding" and (answer.startswith("As an AI assistant") or answer.startswith("I'm sorry,")):
        doc_context = doc_context.split(". ")

    context_size = int(LLAMA_CONTEXT*0.7)
    if LLM_MODEL.startswith("llama") and len(str(answer)) > context_size:
        doc_context = str(answer)[0:context_size]

    return doc_context
    
# Check if local file exists (and contains OBJECTIVE in case of 'task_list.txt')
def check_file(file_name: str):
    try:
        # Check task list output file
        with open(file_name, 'r') as f:
            lines = f.readlines()
            if OBJECTIVE in lines[2]:
                return 'a'
            else:
                return 'w'
    except:
        return 'w'
    

# Write terminal output to 'task_list.txt''
def print_to_file(text: str, mode: chr):
    with open('task_list.txt', mode) as f:
        f.write(text)
# ------------------------
# Function definitions end


# Setup Llama (evaluation and embedding)
if LLM_MODEL.startswith("llama"):
    if can_import("llama_cpp"):
        from llama_cpp import Llama

        print(f"LLAMA : {LLAMA_MODEL_PATH}" + "\n")
        assert os.path.exists(LLAMA_MODEL_PATH), "\033[91m\033[1m" + f"Model can't be found." + "\033[0m\033[0m"

        print('Initialize model for evaluation')
        llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            use_mlock=False,
            seed = -1,  # New parameter
        )
        print('\nInitialize model for embedding')
        llm_embed = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            embedding=True,
            use_mlock=False,
            seed = -1,  # New parameter
        )
        print(
            "\033[91m\033[1m"
            + "\n*****USING LLAMA.CPP. POTENTIALLY SLOW.*****"
            + "\033[0m\033[0m"
        )
    else:
        print(
            "\033[91m\033[1m"
            + "\nLlama LLM requires package llama-cpp. Falling back to GPT-3.5-turbo."
            + "\033[0m\033[0m"
        )
        LLM_MODEL = "gpt-3.5-turbo"

if LLM_MODEL.startswith("gpt-4"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

if LLM_MODEL.startswith("human"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING HUMAN INPUT*****"
        + "\033[0m\033[0m"
    )


# Print objective
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")
print_to_file("\n*****OBJECTIVE*****\n" + f"{OBJECTIVE}\n", mode=check_file('task_list.txt'))
if ENABLE_REPORT_EXTENSION:
    print("\033[93m\033[1m" + "\nAction:" + "\033[0m\033[0m" + f" {ACTION}")
    print_to_file("\nAction:" + f" {ACTION}" + "\n", 'a')
          
if not JOIN_EXISTING_OBJECTIVE:
    print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
    print_to_file("\nInitial task:" + f" {INITIAL_TASK}" + "\n", 'a')
else:
    print("\033[93m\033[1m" + f"\nJoining to help the objective" + "\033[0m\033[0m")
    print_to_file(f"\nJoining to help the objective", 'a')


# Llama embedding function
class LlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        return

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for t in texts:
            e = llm_embed.embed(t)
            embeddings.append(e)
        return embeddings


# Results storage using local ChromaDB
class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.Client(
                settings=chromadb.config.Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=chroma_persist_dir,
                )
            )
        
        metric = "cosine"
        if LLM_MODEL.startswith("llama"):
            embedding_function = LlamaEmbeddingFunction()
        else:
            embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):

        # Break the function if LLM_MODEL starts with "human" (case-insensitive)
        if LLM_MODEL.startswith("human"):
            return
        # Continue with the rest of the function

        embeddings = llm_embed.embed(result) if LLM_MODEL.startswith("llama") else None
        if (
                len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0
        ):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )
        return [item["task"] for item in results["metadatas"][0]]


# Initialize results storage
def try_weaviate():
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_USE_EMBEDDED = os.getenv("WEAVIATE_USE_EMBEDDED", "False").lower() == "true"
    if (WEAVIATE_URL or WEAVIATE_USE_EMBEDDED) and can_import("extensions.weaviate_storage"):
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
        from extensions.weaviate_storage import WeaviateResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Weaviate" + "\033[0m\033[0m")
        return WeaviateResultsStorage(OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_USE_EMBEDDED, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None

def try_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    if PINECONE_API_KEY and can_import("extensions.pinecone_storage"):
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert (
            PINECONE_ENVIRONMENT
        ), "\033[91m\033[1m" + "PINECONE_ENVIRONMENT environment variable is missing from .env" + "\033[0m\033[0m"
        from extensions.pinecone_storage import PineconeResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Pinecone" + "\033[0m\033[0m")
        return PineconeResultsStorage(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None

def use_chroma():
    print("\nUsing results storage: " + "\033[93m\033[1m" + "Chroma (Default)" + "\033[0m\033[0m")
    return DefaultResultsStorage()

results_storage = try_weaviate() or try_pinecone() or use_chroma()


# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]


# Initialize tasks storage
tasks_storage = SingleTaskListStorage()
if COOPERATIVE_MODE in ['l', 'local']:
    if can_import("extensions.ray_tasks"):
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent))
        from extensions.ray_tasks import CooperativeTaskListStorage

        tasks_storage = CooperativeTaskListStorage(OBJECTIVE)
        print("\nReplacing tasks storage: " + "\033[93m\033[1m" + "Ray" + "\033[0m\033[0m")
elif COOPERATIVE_MODE in ['d', 'distributed']:
    pass


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def openai_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.lower().startswith("llama"):
                result = llm(prompt[:LLAMA_CTX_MAX],
                             stop=["### Human"],
                             echo=False,
                             temperature=LLAMA_TEMPERATURE,
                             top_k=40,
                             top_p=0.9,             # default: 0.95
                             repeat_penalty=1.0,    # default: 1.05
                             max_tokens=400)
                # print('\n*****RESULT JSON DUMP*****\n')
                # print(json.dumps(result))
                # print('\n')
                return result['choices'][0]['text'].strip()
            elif model.lower().startswith("human"):
                return user_input_await(prompt)
            elif not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
                # TODO: different limits for different models.

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                messages = [{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


# Create new tasks and store to task list
def task_creation_agent(
        objective: str, result: Dict, task_description: str, task_list: List[str]
):    
    prompt = f"""
You are to use the result from an execution agent to create new tasks with the following objective: {objective}\n
The last completed task has the result: \n{result["data"]}
This result was based on this task description: {task_description}.\n"""

    if task_list:
        prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
    prompt += "Based on the result, return a list of tasks to be completed in order to meet the objective. "
    if task_list:
        prompt += "These new tasks must not overlap with incomplete tasks. "

    prompt += """
Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output.
\nYour response: """

    print(f"\n****TASK CREATION AGENT PROMPT****\n{prompt}")
    print_to_file(f"\n****TASK CREATION AGENT PROMPT****\n{prompt}\n", 'a')
    response = openai_call(prompt, max_tokens=MAX_TOKENS)
    
    print(f"\n****TASK CREATION AGENT RESPONSE****\n{response}")
    print_to_file(f"\n****TASK CREATION AGENT RESPONSE****\n{response}\n", 'a')
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)
            print('New task created: ' + task_name)

    out = [{"task_name": task_name} for task_name in new_tasks_list]
    return out


# Prioritize tasks from task list
def prioritization_agent():
    task_names = tasks_storage.get_task_names()
    bullet_string = '\n'

    prompt = f"""
You are tasked with prioritizing the following tasks: {bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of your team: {OBJECTIVE}.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""

    print(f"\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}")
    print_to_file(f"\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}\n", 'a')

    response = openai_call(prompt, max_tokens=MAX_TOKENS)
    print(f"\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}")
    print_to_file(f"\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}\n", 'a')
    if not response:
        print('Received empty response from priotritization agent. Keeping task list unchanged.')
        print_to_file('Received empty response from priotritization agent. Keeping task list unchanged.', 'a')
        #print('Adding new tasks to task_storage')
        return
    
    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip():
                new_tasks_list.append({"task_id": task_id, "task_name": task_name})
                print('New task created: ' + task_name)

    return new_tasks_list


# Execute a task based on the objective and context
def execution_agent(objective: str, task: str) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """
    context = context_agent(objective, task, top_results_num=5)
    doc_context = ""
    if ENABLE_EMBEDDINGS_EXTENSION:
        doc_context = qa_retrieval(task, context, "embedding")
    if WIKI_CONTEXT:
        wiki_context = qa_retrieval(task, context, "wiki")

    prompt = f'Perform one task based on the following objective: {OBJECTIVE}\nYour task: {task}\n'
    if ENABLE_REPORT_EXTENSION and INITIAL_TASK not in task:
        prompt += f'Consider the following action which shall be executed based on the objective: {ACTION}\n'
    if context:
        prompt += 'Take into account these previously completed tasks:' + '\n'.join(context)
    if ENABLE_EMBEDDINGS_EXTENSION and doc_context != "":
        prompt += f'\nConsider the answer on the task from related document embedding query: {doc_context}'
    elif WIKI_CONTEXT and wiki_context:
        prompt += f'\nConsider the answer on the task from wikipedia search query: {wiki_context}'
    prompt += '\nIgnore implausible and incomprehensible information. Do not add any notes related to you being an AI assistant.'
    if ENABLE_SEARCH_EXTENSION:
        #prompt += f'\nDo your best to complete the task and provide a useful answer. Responding with the task content itself, a variation of it or vague suggestions, is not useful as well. In this case assume that internet search is required.'
        if ENABLE_EMBEDDINGS_EXTENSION:
            prompt += '\nIf internet search will most probably help to complete the task, add "Internet search request: " to the response and add the task redrafted to an optimal concise internet search request.'
        else:
            prompt += '\nIf internet search is required to complete the task, add "Internet search request: " to the response and add in the same line the task redrafted to an optimal concise internet search request.'
    if WIKI_SEARCH:
        prompt += '\nIf wikipedia search is suited best to complete the task, add "Wikipedia search request: " to the response and add in the same line the task redrafted to an optimal concise wikipedia search request.'
    prompt += f'\n\nYour response: '
    return openai_call(prompt, max_tokens=MAX_TOKENS), doc_context


# Get the top n completed tasks for the objective
def context_agent(objective: str, task: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    #query = objective   #task + ". Take into account the ultimate objective: " + objective
    query = f'Give advice for a query, consisting of an ultimate objective and a task.'
    query += f'\nObjective: {OBJECTIVE}'
    query += f'\nTask: {task}'
    query += f'\n\nYour response: '

    results = results_storage.query(query=query, top_results_num=top_results_num)
    print(f"\033[96m\033[1m\n*****RELEVANT CONTEXT*****\033[0m\033[0m\n{results}")
    print_to_file(f"\n*****RELEVANT CONTEXT*****\n{results}\n", 'a')
    # print("****RESULTS****")
    # print(results)
    return results


# Check if google needs to be accessed, based on the the text in last completed task result
def internet_agent(result: str, task: str):
    search_request = ""
    search_results = ""
    scrape_page = ""
    scrape_raw = ""
    links = []
    wiki_flag = False

    # Extraction of search request from task result text
    if "search request: " or "search query: " in result:
        if "search query: " in result:
            line = result.split("search query: ")[1]
            if line:
                search_request = line
            if ("Wiki" or "wiki") in result:
                wiki_flag = True

        elif "search request: " in result:
            line = result.split("search request: ")[1]
            if line:
                search_request = line
            if ("Wiki" or "wiki") in result:
                wiki_flag = True

        # Substitue for search request
        if search_request != "":
            print("\nExtracted search request: " + search_request)
            if INITIAL_TASK in search_request:
                if INITIAL_TASK not in task:
                    search_request = task
                else:
                    search_request = INITIAL_TASK + " for objective: " + OBJECTIVE
                print("Substituted search request (inital task in search request): " + search_request)
            elif search_request == "":
                if task:
                    search_request = task
                else:
                    search_request = OBJECTIVE
                print("Substituted search request (search request is empty): " + search_request)

            num_results = 3     # 5 is better, but web scrape result summarization is token intensive and may exceed context size
            if LLM_MODEL.startswith("llama"):
                num_results = 1
        
        # a.) Get top urls from smart search and scrape web pages with LLM powered summarization
        if not wiki_flag and search_request != "":
            if GOOGLE_API_KEY and GOOGLE_CSE_ID:
                print("Access smart search with Google CSE...")
                print_to_file("\nAccess smart search with Google CSE...\n", 'a')
                search_results, scrape_page, scrape_raw, links = web_search_tool(query=search_request, task=task, num_extracts=num_results, mode="google", summary_mode=True)
            elif SERPAPI_API_KEY:
                print("Access smart search with SERPAPI...")
                print_to_file("\nAccess smart search with SERPAPI...\n", 'a')
                search_results, scrape_page, scrape_raw, links = web_search_tool(query=search_request, task=task, num_extracts=num_results, mode="serpapi", summary_mode=True)
            else:
                print("Access smart search with www.duckduckgo.com...")
                print_to_file("\nAccess smart search with www.duckduckgo.com...\n", 'a')
                search_results, scrape_page, scrape_raw, links = web_search_tool(query=search_request, task=task, num_extracts=num_results, mode="browser", summary_mode=True)

        # b.) Access wikipedia API
        elif WIKI_SEARCH and wiki_flag and search_request != "":
            print("Accessing wikipedia API...")
            search_results = wikipedia.run(search_request)
            links = ['https://en.wikipedia.org/wiki/']

    # Filter out irrelevant parts from summary
    if search_results != "":
        if "No relevant information found in the text to be analyzed.. " in search_results:
            search_results = search_results.replace("No relevant information found in the text to be analyzed.. ", "")
        elif "No relevant information found in the text.. " in search_results:
            search_results = search_results.replace("No relevant information found in the text.. ", "")
        elif ".." in search_results:
            search_results = search_results.replace("..", "")
        if len(search_results) < 10:
            search_results = ""
        
        # Filter out irrelevant results for complete web pages
        for line in search_results.split("\n"):
            if line.startswith("I am sorry") or line.startswith("I'm sorry") or line.startswith("Sorry") or line.startswith("I apologize") or line.startswith("I cannot"):
                search_results = search_results.replace(line, "")             
        result = search_results
    else:
        result = str("")

    if result != "":
        context_size = int(LLAMA_CONTEXT*0.9)
        if LLM_MODEL.startswith("llama") and len(result) > context_size:
            result = result[0:context_size]
        print("\033[93m\033[1m" + "\n*****TASK RESULT WITH SMART SEARCH*****" + "\033[0m\033[0m\n" + result)
        print_to_file("\n*****TASK RESULT WITH SMART SEARCH*****\n" + result + "\n", 'a')
        print(f'\nWeb scrape content (filtered HTML extract, w/o style sheets, scripts and filtered for tags):\n{scrape_raw}\n')

    return result, search_results, scrape_page, scrape_raw, search_request, str(links)


# Store task result to memory file
def store_result(result: str, task: str):
    if ("task list" and "Task list" and "Task List") not in result:
        if INITIAL_TASK in task:
            task_name = f"{INITIAL_TASK} for objective: {OBJECTIVE}"
        elif task != "":
            task_name = task
        else:
            task_name = OBJECTIVE
        
        if result.startswith("As an AI assistant"):
            result = result.split(". ")

        qa_memory = f'\n--------\nQuestion: {task_name}\nAnswer: {result}\n'
        file_path = SCRAPE_SOURCE_PATH + "/qa_memory.txt"
        print()
        text_writer(file_path=file_path, input=qa_memory, text="task result")
    else:
        qa_memory = ""
    return qa_memory


# Add the initial task if starting new objective
if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {
        "task_id": tasks_storage.next_task_id(),
        "task_name": INITIAL_TASK
    }
    tasks_storage.append(initial_task)

# Trigger initial smart search and setup document embedding vector store with results
if ENABLE_SEARCH_EXTENSION and INITIAL_SEARCH:
    initial_search(search_request=OBJECTIVE, task="", num_results=5, num_requests=5)


def main():
    loop = True
    while loop:
        # As long as there are tasks in the storage...
        if not tasks_storage.is_empty():
            # Print the task list
            print("\033[95m\033[1m" + "\n*****TASK LIST*****" + "\033[0m\033[0m")
            print_to_file("\n*****TASK LIST*****\n", 'a')
            for t in tasks_storage.get_task_names():
                print(" • " + str(t))
                print_to_file((" • " + str(t)), 'a')
            print_to_file("\n", 'a')

            # Step 1: Pull the first incomplete task
            task = tasks_storage.popleft()
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****" + "\033[0m\033[0m")
            print(str(task["task_name"]))
            print_to_file("\n*****NEXT TASK*****\n" + str(task["task_name"]) + "\n", 'a')

            # Send to execution function to complete the task with given objective
            result = ""
            doc_result = ""
            qa_result = ""
            result, doc_result = execution_agent(OBJECTIVE, str(task["task_name"]))
            print("\033[93m\033[1m" + "\n*****TASK RESULT*****" + "\033[0m\033[0m\n" + result)
            print_to_file("\n*****TASK RESULT*****\n" + result + "\n", 'a')
            if EMBEDDINGS_BACKUP:
                qa_result = store_result(result, str(task["task_name"]))
            if ENABLE_REPORT_EXTENSION:
                check_report(result)

            # Step 2: Check if internet search is required for conclusion of the task
            summary_result = ""
            search_request = ""
            search_result = ""
            page_raw = ""
            page_text = ""
            links = ""
            scrape_summary = ""
            if ENABLE_SEARCH_EXTENSION or WIKI_SEARCH:
                summary_result, search_result, page_text, page_raw, search_request, links = internet_agent(result, str(task["task_name"]))
                if search_result != "":
                    if ENABLE_REPORT_EXTENSION:
                        check_report(search_result)

                    # Add all available internet results to file (header info is added for scrape_summary and scrape_validated)
                    if EMBEDDINGS_BACKUP:
                        if search_result != "":
                            scrape_summary = f'\nInternet search request: {search_request}\n\nWeb scrape result summary:\n{search_result}\n\nSources: {links}\n'
                            scrape_raw = f'\nWeb scrape content:\n{page_raw}\n\nSources: {links}'
                            file_path = SCRAPE_SOURCE_PATH + "/qa_memory.txt"
                            text_writer(file_path=file_path, input=scrape_summary, text="scape result summary")
                            text_writer(file_path=file_path, input=scrape_raw, text="scrape raw content")

            # Update document embeddings vectorstore with all available results, if web scrape result is available
            if EMBEDDINGS_UPDATE and search_result != "":
                print("\033[93m\033[1m" + "\n*****RESULTS EMBEDDING IN DOCUMENT VECTORSTORE*****" + "\033[0m\033[0m")
                print_to_file("\n*****RESULTS EMBEDDING IN DOCUMENT VECTORSTORE*****\n", 'a')
                scrape_summary = f'\nInternet search request: {search_request}\n\nWeb scrape result summary:\n{search_result}\n'
                scrape_raw = f'\nWeb scrape content:\n{page_raw}\n'
                input = f'{qa_result}{scrape_summary}{scrape_raw}'
                res = text_loader(DOC_STORE_NAME, input, links)
                if res:
                    db, retriever = update_db()

            # Llama failsafe: Check for result supplement
            if summary_result != "":
                result += f'\n\n{summary_result}'
            elif doc_result != "":
                result += f'\n\n{doc_result}'

            # Step 3: Enrich result and store in the results storage
            # This is where you should enrich the result if needed
            enriched_result = {
                "data": result
                #"doc": doc_result,
                #"web_summary": summary_result,
                #"web_scrape": page_raw
            }
            # extract the actual result from the dictionary
            # since we don't do enrichment currently
            #vector = result + "\n\n" + doc_result + "\n\n" + summary_result + "\n\n" + page_raw
            result_id = f"result_{task['task_id']}"
            results_storage.add(task, result, result_id)  

            # Step 4: Create new tasks and re-prioritize task list
            # only the main instance in cooperative mode does that
            new_tasks = task_creation_agent(
                OBJECTIVE,
                enriched_result,
                task["task_name"],
                tasks_storage.get_task_names(),
            )
            
            print('Adding new tasks to task_storage...')
            for new_task in new_tasks:
                new_task.update({"task_id": tasks_storage.next_task_id()})
                print(str(new_task))
                tasks_storage.append(new_task)

            if not JOIN_EXISTING_OBJECTIVE:
                prioritized_tasks = prioritization_agent()
                if prioritized_tasks:
                    tasks_storage.replace(prioritized_tasks)

            # Sleep a bit before checking the task list again
            time.sleep(5)

        else:
            print('Done.')
            loop = False


if __name__ == "__main__":
    main()

