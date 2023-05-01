#!/usr/bin/env python3
import os
import subprocess
import time
from collections import deque
from typing import Dict, List
import importlib
import openai
import pinecone
from dotenv import load_dotenv, set_key

from extensions.google_search import get_toplist
from datetime import datetime, timedelta


# Load default environment variables (.env)
load_dotenv()

# Engine configuration

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
assert (
    PINECONE_ENVIRONMENT
), "PINECONE_ENVIRONMENT environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Parameter configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
STOP_CRITERIA = os.getenv("STOP_CRITERIA", "")
SAFETY_THRESHOLD = os.getenv("SAFETY_THRESHOLD", "")
PROBABILITY_THRESHOLD = os.getenv("PROBABILITY_THRESHOLD", "")
PROBABILITY_HIGHSCORE = os.getenv("PROBABILITY_HIGHSCORE", "")
CONTRIBUTION_THRESHOLD  = os.getenv("CONTRIBUTION_THRESHOLD", "")
STORED_CONTRIBUTION = os.getenv("STORED_CONTRIBUTION", "")
STORED_PLAUSIBILITY = os.getenv("STORED_PLAUSIBILITY", "")
STORED_PROBABILITY = os.getenv("STORED_PROBABILITY", "")
STORED_HIGHSCORE = os.getenv("STORED_HIGHSCORE", "")
STORED_RUNTIME = os.getenv("STORED_RUNTIME", "")
FINAL_PROMPT = os.getenv("FINAL_PROMPT", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))

# Google API configuration
YOUR_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
YOUR_SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")


# Extensions support begin

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

        OBJECTIVE, INITIAL_TASK, OPENAI_API_MODEL, DOTENV_EXTENSIONS = parse_arguments()

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions # but also provide command line
# arguments to override them

# Extensions support end

# Check if we know what we are doing
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"
assert INITIAL_TASK, "INITIAL_TASK environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )


# Check initial OBJECTIVE in task_list.txt
def check_file():
    try:
        with open('task_list.txt', 'r') as f:
            lines = f.readlines()
            for l in lines:
                if OBJECTIVE in l:
                    return 'a', float(STORED_CONTRIBUTION), float(STORED_PLAUSIBILITY), float(STORED_PROBABILITY), int(STORED_HIGHSCORE), STORED_RUNTIME
                else:
                    return 'w', 0.0, 0.0, 0.0, 0, "00:00:00"
    except:
        return 'w', 0.0, 0.0, 0.0, 0, "00:00:00"
        

# Write output to file
def write_to_file(text: str, mode: chr):
    with open('task_list.txt', mode) as f:
        f.write(text)


# Read-in trigger.txt (for stopping the program and triggering the final prompt)
def check_trigger():
    try:
        with open('trigger.txt', 'r') as f:
            lines = f.readlines()
            if (lines[0].strip() == 'FINAL STOP'):
                return "FINAL STOP"
            if (lines[0].strip() == 'STOP'):
                return "STOP"
            else:
                return "NO TRIGGER"
    except:
        return "ERROR"
    

# Check if Google custom search API keys are setup
def check_google_keys():
    if YOUR_GOOGLE_API_KEY and YOUR_SEARCH_ENGINE_ID:
        return True
    else:
        return False


# Counter for calculation of overall achievement data (in percentage*0.01) and runtime
contribution_counter = 0.0
plausibility_counter = 0.0
probability_counter = 0.0
probability_highscore = 0   # highscore as integer (number of task >= PLAUSIBILITY_THRESHOLD)
mode, contribution_counter, plausibility_counter, probability_counter, probability_highscore, stored_runtime = check_file()
hours, minutes, seconds = map(int, stored_runtime.split(':'))
runtime_seconds = timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()
start_time = time.time() - runtime_seconds

# Clear evaluation counters in environment file
if mode == 'w':
    set_key(".env", "STORED_CONTRIBUTION", "0.0")
    set_key(".env", "STORED_PLAUSIBILITY", "0.0")
    set_key(".env", "STORED_PROBABILITY", "0.0")
    set_key(".env", "STORED_HIGHSCORE", "0.0")
    set_key(".env", "STORED_RUNTIME", "00:00:00")

# Print OBJECTIVE, STOP_CRITERIA, INITIAL_TASK and setup plausibilization/contribution variables
write_to_file(f"*****OBJECTIVE*****\n{OBJECTIVE}\n\n*****STOP CRITERIA*****\n{STOP_CRITERIA}\n\nInitial task: {INITIAL_TASK}\n", mode)
print(f"\033[94m\033[1m\n*****OBJECTIVE*****\n\033[0m\033[0m{OBJECTIVE}")
print(f"\033[91m\033[1m\n*****STOP CRITERIA*****\n\033[0m\033[0m{STOP_CRITERIA}")
print(f"\033[93m\033[1m\nInitial task:\033[0m\033[0m {INITIAL_TASK}")

# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index
table_name = YOUR_TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(
        table_name, dimension=dimension, metric=metric, pod_type=pod_type
    )

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])


def add_task(task: Dict):
    task_list.append(task)


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 200,
):
    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return result.stdout.strip()
            elif not model.startswith("gpt-"):
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
                # Use chat completion API
                messages = [{"role": "system", "content": prompt}]
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
                "   *** OpenAI API timeout occured. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occured. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occured. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
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


# Create new tasks with reasoning for stop criteria and calculation of last task's result contribution to objective
def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""
    You are a task creation AI that uses the result of the last completed task to create new tasks with the objective: {objective}
    The last completed task has the result: {result}, based on this task description: {task_description}
    Incomplete tasks: {', '.join(task_list)}.
    Take into account the stop criteria: {STOP_CRITERIA}

    Create new tasks without overlapping with incomplete tasks and with as few tasks as possible to achieve the objective, and consider that they are for a large language model as yourself.

    Take into account the stop criteria. If it is met, create a new task with only content 'Stop criteria has been met...'.

    Return all the new tasks, with one task per line in your response. Do not follow your response with any other output. Do not display information from metadata in output.
    
    The result must be a numbered list in the format:
    #. First task
    #. Second task
    #. ...
    """
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]


# Prioritize the task list
def prioritization_agent(this_task_id: int, threshold: float):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    prompt = f"""
    You are an AI that cleans and prioritizes tasks based on their relevance to the objective: {OBJECTIVE}

    Tasks: {task_names}
    Next task ID: {next_task_id}
    Threshold: {threshold}

    Tasks should be sorted from highest to lowest priority. 
    Higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
    If the last completed task's contribution is below {threshold} and clear, suggest new tasks in a different subject area.

    Consider the task metadata and prioritize tasks to meet the objective efficiently, taking into account task dependencies, task relevance and other factors derived from the metadata:

    1. "contribution": level of contribution of the task to the objective (0.0 to 1.0)
    2. "difficulty": level of difficulty of the task (0.0 to 1.0)
    3. "plausibility": level of plausibility of the task with respect to other information available (0.0 to 1.0)
    4. "probability": level of probability of the task to achieve the objective (0.0 to 1.0)
    5. "num_tokens": number of tokens for the response to the task
    6. "categories": categories regarding the topics in content of task
    7. "keywords": keywords derived from content of the task
    
    Do not remove tasks. Rearrange tasks as needed.
    
    Return a numbered list starting with {next_task_id}:
    #. First task
    #. Second task
    #. ...

    The entries are consecutively numbered, starting with 1. The number of each entry must be followed by a period.
    Do not include any headers before your numbered list. Do not follow your numbered list with any other output.
    Do not display information from metadata in numbered list. If metadata information is included in the the content of a task, delete it.
    """
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


# Execute a task based on the objective and five previous tasks, verbalize the task to a request for internet search if required
def execution_agent(objective: str, task: str, internet: bool) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.
        internet (bool): Whether the AI should perform internet research for the task.

    Returns:
        str: The response generated by the AI for the given task.

    """
    context = context_agent(query=objective, top_results_num=5)
    if task == INITIAL_TASK and initial_plan != "":
        context.append(initial_plan)

    if internet == True:
        print(f"\033[93m\033[1m\n*****TASK RESULT (WITH INTERNET RESEARCH)*****\033[0m\033[0m")
        write_to_file(f"\n*****TASK RESULT (WITH INTERNET RESEARCH)*****\n", 'a')
    else:
        print(f"\033[96m\033[1m\n*****RELEVANT CONTEXT*****\033[0m\033[0m\n{context}")
        write_to_file(f"*****RELEVANT CONTEXT*****\n{context}\n", 'a')

    prompt = f"""
    You are a task execution AI performing one task based on the following objective: {objective}\n 
    Consider these previously completed tasks: {context}.\n
    Your task: {task}

    Do your best to complete the task, which means to provide a useful answer for a human, responding with the task content itself, a variation of it or vague suggestions, is not useful as well.
    In case you are not able to determine an useful answer, assume that internet search is required and verbalize the reason.

    First, respond as instructed above.

    Next, if the task is not completed or internet search is required to complete the task, respond with 'Internet search request: ' and redraft the task to an optimal short internet search request.
    """
    return openai_call(prompt, max_tokens=2000)


# Get the top n completed tasks for the objective
def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    query_embedding = get_ada_embedding(query)
    results = index.query(query_embedding, top_k=top_results_num, include_metadata=True, namespace=OBJECTIVE)
    #print(f"\033[96m\033[1m\n*****CONTEXT QUERY*****\n\033[0m\033[0m{results}")
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]


# Get the contribution of the last completed task result
def result_eval_agent(result: str, contribution_counter: float, plausibility_counter: float, probability_counter: int, probability_highscore: int):
    prompt = f"""
    You are an AI that evaluates the contribution of a task result to the objective.

    Task result: {result}
    Objective: {OBJECTIVE}

    If the task result includes a list of tasks, determine the following values for the list as a whole and respond for the whole list.

    First, determine the task result's contribution to the objective, ranging from 0.0 (no contribution) to 1.0 (objective achieved), without any unit.

    Respond with 'Contribution: ' followed by the number if the contribution is clear, otherwise 'Contribution: unclear' and do not respond with anything else.

    Next, determine the rating of the task result, ranging from 0.0 (very easy) to 1.0 (very difficult), without any unit.
    Try your best to evalaute a rating and assign a number even or greater than 0.0.

    Respond with 'Difficulty: ' followed by the number if the difficulty is clear, otherwise 'Difficulty: unclear' and do not respond with anything else.

    Next, determine the plausibility of the task result, with respect to other information available, ranging from 0.0 (very unplausible) to 1.0 (very plausible), without any unit. If there's any plausibility at all, assign a number greater than 0.0.

    Respond with 'Plausibility: ' followed by the number if the plausibility is clear, otherwise 'Pausibility: unclear' and do not respond with anything else.

    Next, determine the task result's probability to meet the objective. The range is from 0.0 (very far away) to 1.0 (objective met), without any unit.

    Respond with 'Probability: ' followed by the number if the probability is clear, otherwise 'Probability: unclear'. And do not respond with anything else.

    Next, determine the up to 3 most relevant categories regarding the task result.

    Respond with 'Categories: ' followed by the categories.

    Next, determine up to 5 most relevant keywords of the task result.

    Respond with 'Keywords: ' followed by the keywords.
    """
    response = openai_call(prompt)
    lines = response.split("\n")
    contribution = -2.0
    difficulty = -2.0
    plausibility = -2.0
    categories = ""
    keywords = ""
    probability = -2.0
    for l in lines:
        # Get the contribution value
        if "Contribution" in l:
            try:
                contribution = float(l.split(": ")[1])
            except (ValueError, IndexError):
                contribution = -1.0

        # Get the difficulty value
        if "Difficulty" in l:
            try:
                difficulty = float(l.split(": ")[1])
            except (ValueError, IndexError):
                difficulty = -1.0

        # Get the plausibility value
        if "Plausibility" in l:
            try:
                plausibility = float(l.split(": ")[1])
            except (ValueError, IndexError):
                plausibility = -1.0

        # Get the probability value
        if "Probability" in l:
            try:
                probability = float(l.split(": ")[1])
            except (ValueError, IndexError):
                probability = -1.0

        # Get the categories
        if "Categories" in l:
            try:
                categories = l.split(": ")[1]
            except (ValueError, IndexError):
                categories = str("")

        # Get the keywords
        if "Keywords" in l:
            try:
                keywords = l.split(": ")[1]
            except (ValueError, IndexError):
                keywords = str("")
        #print(f"\nl: {l}")

    # Update evaluation counters
    if contribution > 0 and contribution <= 100:
        contribution_counter += contribution
        set_key(".env", "STORED_CONTRIBUTION", str(contribution_counter))

    if plausibility > 0 and plausibility <= 100:
        plausibility_counter += plausibility
        set_key(".env", "STORED_PLAUSIBILITY", str(plausibility_counter))

    if probability > 0 and probability <= 100:
        probability_counter += probability
        set_key(".env", "STORED_PROBABILITY", str(probability_counter))

    if probability >= float(PROBABILITY_THRESHOLD):
        probability_highscore += 1
        set_key(".env", "STORED_HIGHSCORE", str(probability_highscore))
    
    # Print metadata to terminal
    print(f"\n{response}")
    #write_to_file(f"{response}\n", 'a')

    # Debug output
    #print(f"Probability Value: {probability}")
    #print(f"\nLines: {lines}")
    return contribution, difficulty, plausibility, probability, probability_highscore, categories, keywords, contribution_counter, plausibility_counter, probability_counter
    

# Assess the objective, stop criteria, and final prompt
def assess_objective():
    prompt = f"""
    You are an AI that evaluates the objective and stop criteria, used for a task-based AI as goal definitions, and the final prompt which is triggered when the task-based process is finished.

    Objective: {OBJECTIVE}
    Stop criteria: {STOP_CRITERIA}
    Final prompt: {FINAL_PROMPT}

    As a human, estimate the following, in short and concisive manner:
    1. Probability of achieving the objective (0-100%).
    2. Time to research and complete the task list (in hours and minutes).

    For the following items, also consider that these are prompts for a large language model like yourself, and make drafts:
    1. Propose an optimal the stop criteria for best results with reasonable completion time.
    2. Propose an optimal softened version of the stop criteria, considering the task-based AI approach.
    3. Propose an optimal objective for best results, considering the task-based AI approach.
    4. Propose an optimal updated version of the final prompt considering the task-based AI approach and the objective.
    """
    return openai_call(prompt, max_tokens=200)


# Execute the final task
def final_response(objective: str, task: str, num_token: int) -> str:
    context = context_agent(query=objective, top_results_num=30)
    prompt = f"""
    You are a task execution AI performing one task based on the following objective: {objective}

    Consider these previously completed tasks and include relevant information in the response: {context}

    Your task: {task}

    Response: """
    return openai_call(prompt, max_tokens=num_token)


# Send final prompt and handle final response(s)
def final_prompt():
    response = final_response(OBJECTIVE, FINAL_PROMPT, num_token=2000)
    write_to_file(f"*****FINAL RESPONSE*****\n{response}\n", 'a')
    print(f"\033[94m\033[1m\n*****FINAL RESPONSE*****\n\033[0m\033[0m{response}")
    while "The final response generation has been completed" not in response:
        response = openai_call("Continue. Respond with 'The final response generation has been completed...', and nothing else, in case the complete result has been responded already or the question is not understood.")
        write_to_file(f"{response}", 'a')
        print(response)
    write_to_file("\n\n", 'a')
    print("\n***** The objective has been achieved, the work is done! BabyAGI will take a nap now... *****\n")


# Send initial prompt with the objective, as guidance for INITIAL_TASK
def inital_request():
    response = final_response(OBJECTIVE, FINAL_PROMPT, num_token=1000)
    prompt = f"""
    You are an AI that determines supporting information to support a following task-based AI in setting up a task list for the objectice.
    This is the objective: {OBJECTIVE}

    Consider the given response on the objective, revealing relevant information, related topics, subject areas for research, answers for the objective, difficulties in finding an answer, and more. Determine the relevant information. Do not compile a task list, but suggest supporting information for the task-based AI to compile an optimal task list.
    This is the response: {response}

    Consider that the supporting information is for a large language model like yourself and verbalize in short and concisive manner.

    Determine supporting information to optimally support the creation of a task list by the task-based AI.

    Respond with the initial plan: 
    """
    return openai_call(prompt, max_tokens=500)


# Provide internet access via Google API, using google_search.py, and summary of results from snippets
# max value for num_results is 10 (results per page)
def internet_research(topic: str, num_results, num_pages, num_extracts: int):
    toplist_result, page_content, links = get_toplist(topic, YOUR_GOOGLE_API_KEY, YOUR_SEARCH_ENGINE_ID, num_results, num_pages, num_extracts)
    if toplist_result == []:
        toplist_result = "\n *** No data returned from Google custom search API... ***"

    print(f"\nGoogle search top results:\n{str(toplist_result)}")
    for i in range(num_extracts):
        print(f"\nWebpage content ({str(links[i])}):\n{str(page_content[i])}")
    return toplist_result, page_content


# Check if google needs to be accessed, based on the the text in last completed task result
def check_search_request(result: str):
    lines = result.split("\n")
    search_request = str("")
    #print(f"All lines: {lines}\n")
    for l in lines:
        #print(f"Line: {l}")
        if "Internet search request:" in l:
            search_request = l.split(": ")[1]
            #print(f"Request: {search_request}")
        elif (str(task["task_name"]) or "researching" or "consulting" or "I will" or "I'm sorry, " or "need more information") and "Keywords:" in l and ("Task List:" or "Task list:" or "task list:") not in l:
            search_request = l.split("Keywords: ")[1].split(")")[0]
        else:
            search_request = str("")

    # Check if google needs to and can be accessed
    if search_request and check_google_keys():
        print(f"{result}")
        write_to_file(f"{result}\n", 'a')
        print("Accessing Google custom search API...")
        write_to_file(f"Accessing Google custom search API...\n", 'a')
        num_extracts = 3
        toplist, webpages = internet_research(str(search_request), num_results=10, num_pages=1, num_extracts=3)
        updated_task = "This is the previous task: " + str(task["task_name"]) + "\nInternet research has been performed for the previous task with the following request: " + search_request + f"\nResult from Google search top list: {str(toplist)}\nContent of top {num_extracts} webpage: {webpages}\nTake into account that not all the information from internet research might be relevant for the task. Evaluate which information is relevant and which is not and complete the task accordingly, considering also other information available than the internet research, and respond again on the previous task."
        #print(f"\nUpdated Task:\n{str(updated_task)}")
        result = execution_agent(OBJECTIVE, str(updated_task), True) 
    return result


# Evaluate feasibility for the objective with given stop criteria and make proposals for optimization
text = assess_objective()
print(f"\n\033[90m\033[1m*****FEASIBILITY EVALUATION*****\033[0m\033[0m\n{text}")
#write_to_file(f"*****FEASIBILITY EVALUATION*****\n{text}\n", 'a')

# Add the first task
first_task = {"task_id": 1, "task_name": INITIAL_TASK}
add_task(first_task)
# Main loop
task_id_counter = 1
initial_plan = ""
initial_plan = inital_request()
print(f"\033[94m\033[1m\n*****INITAL PLAN*****\n\033[0m\033[0m{initial_plan}")
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****" + "\033[0m\033[0m")
        write_to_file("\n*****TASK LIST*****\n", 'a')
        for t in task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])
            write_to_file(str(t["task_id"]) + ": " + t["task_name"] + "\n\n", 'a')

        # Step 1: Pull the first task
        task = task_list.popleft()

        # Step 2: Check for stop criterias (LLM reasoning only, or with additional confidence threshold or by manual trigger)
        if ("Stop criteria has been met" in task["task_name"] or "FINAL STOP" in check_trigger()) or (plausibility_counter >= float(SAFETY_THRESHOLD) and contribution_counter >= float(SAFETY_THRESHOLD) and probability_counter >= float(SAFETY_THRESHOLD) and float(SAFETY_THRESHOLD) > 0.0) or (probability_highscore > int(PROBABILITY_HIGHSCORE) and int(PROBABILITY_HIGHSCORE) > 0):
            final_prompt()
            break

        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m" + str(task["task_id"]) + ": " + task["task_name"])
        write_to_file("*****NEXT TASK*****\n" + str(task["task_id"]) + ": " + task["task_name"] + "\n\n", 'a')

        # Step 3: Send task to execution agent to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"], False)
        this_task_id = int(task["task_id"])
        print(f"\033[93m\033[1m\n*****TASK RESULT*****\033[0m\033[0m")
        write_to_file(f"\n*****TASK RESULT*****\n", 'a')

        # Step 4: Check if internet search is required for conclusion of this task (only when Google API key and search engine ID are provided) 
        result = check_search_request(result)            
        print(f"{result}")
        write_to_file(f"{result}\n", 'a')

        # Step 5: Enrich result with metadata and store in Pinecone, after contribution value for the task result has been calculated
        contribution, difficulty, plausibility, probability, probability_highscore, categories, keywords, contribution_counter, plausibility_counter, probability_counter = result_eval_agent(result, contribution_counter, plausibility_counter, probability_counter, probability_highscore)
        num_tokens = len(result.split())
        enriched_result = {
            "data": result,
            "contribution": str(contribution),
            "difficulty": str(difficulty),
            "plausibility": str(plausibility),
            "probability": str(probability),
            "categories": categories,
            "keywords": keywords,
            "num_tokens": str(num_tokens)
        }  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = get_ada_embedding(
            enriched_result["data"]
        )  # get vector of the actual result extracted from the dictionary
        index.upsert(
            [(result_id, vector, {
            "task": task["task_name"],
            "result": result,
            "contribution": str(contribution),
            "plausibility": str(plausibility),
            "probability": str(probability),
            "categories": str(categories),
            "keywords": str(keywords),
            "num_tokens": str(num_tokens)})],
	        namespace=OBJECTIVE
        )

        # Calculate runtime
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        timestamp = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        set_key(".env", "STORED_RUNTIME", timestamp)

        # Print the enriched results
        print(f"\033[94m\033[1m\n*****STATUS EVALUATION*****\033[0m\033[0m") 
        print(f"Probability high score (>={float(PROBABILITY_THRESHOLD)*100}%): {probability_highscore} with threshold: {int(PROBABILITY_HIGHSCORE)}")
        print(f"Contribution counter: {contribution_counter:.2f}, Plausi counter: {plausibility_counter:.2f}, Probability counter: {probability_counter:.2f} with threshold: {SAFETY_THRESHOLD}")
        print(f"BabyAGI runtime [{timestamp}]")
        #write_to_file(f"\n*****STATUS EVALUATION*****\nContribution counter: {contribution_counter:.1f}, Plausibility counter: {plausibility_counter:.2f}, Probability counter: {probability_counter:.2f} with threshold: {SAFETY_THRESHOLD} and BabyAGI runtime [{timestamp}]\n", 'a')
        #write_to_file(f"Probability high score (>={float(PROBABILITY_THRESHOLD)*100}%): {probability_highscore} with threshold: {int(PROBABILITY_HIGHSCORE)}\n", 'a')

        # Check for manual program stop trigger
        if check_trigger() == "STOP":
            write_to_file("\n\n", 'a')
            print("\n***** BabyAGI has been stopped by manual trigger... *****\n")
            break

        # Step 6: Create new tasks, reprioritize task list and calculate contribution value for last task result
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list]
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id, CONTRIBUTION_THRESHOLD)

    time.sleep(1)  # Sleep before checking the task list again
    
