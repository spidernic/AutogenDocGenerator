"""
Module: code_documentation_generator.py
Description:
    This script analyzes code repositories and generates comprehensive documentation
    including purpose, functionalities, architecture, notable features, libraries used,
    and usage examples. It utilizes OpenAI's language models via the AutoGen framework
    and includes enhanced error handling, concurrency management, and logging.

## Author Information
- **Author**: Nic Cravino
- **Email**: spidernic@me.com 
- **LinkedIn**: [Nic Cravino](https://www.linkedin.com/in/nic-cravino)
- **Date**: October 26, 2024
- **UPDate**: November 10, 2024 - refactored agent creation
Explanation of Changes:
    Direct Initialization: The initialize_agents() function now directly initializes and returns the code_analyzer, documentation_assistant, and user_proxy agents without the need for a dedicated AgentManager class.
    Agent Usage: In main(), the initialize_agents() function is called to set up the agents, which are then passed into the CodeAnalyzer and RepositoryProcessor classes as before.

## License: Apache License 2.0 (Open Source)
This tool is licensed under the Apache License, Version 2.0. This is a permissive license that allows you to use, distribute, and modify the software, subject to certain conditions:

- **Freedom of Use**: Users are free to use the software for personal, academic, or commercial purposes.
- **Modification and Distribution**: You may modify and distribute the software, provided that you include a copy of the Apache 2.0 license and state any significant changes made.
- **Attribution**: Original authorship and contributions must be acknowledged when redistributing the software or modified versions of it.
- **Patent Grant**: Users are granted a license to any patents covering the software, ensuring protection from patent claims on original contributions.
- **Liability Disclaimer**: The software is provided "as is," without warranties or conditions of any kind. The authors and contributors are not liable for any damages arising from its use.

For full details, see the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
"""

import os
import json
import logging
import time
import random
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import xml.etree.ElementTree as ET
from jsonschema import validate, ValidationError
import threading
import pkg_resources
import yaml  # For configuration management
import openai

# Load environmental variables securely
load_dotenv(override=True)
api_key = os.getenv('OPEN_AI_API_KEY')
model_name = os.getenv('MODEL_NAME')

# Load configurations from config.yaml
try:
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    config = {}

semilla = config.get('semilla', 42)
temperature = config.get('temperature', 0.2)
max_api_calls_per_minute = config.get('max_api_calls_per_minute', 60)
calls_per_second = max_api_calls_per_minute / 60
max_workers = int(calls_per_second)
repo_path = config.get('repo_path', "AutogenMermaid")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Rate limiting mechanism
rate_lock = threading.Lock()
last_time_called = [0.0]

def rate_limited_call(func):
    def wrapper(*args, **kwargs):
        with rate_lock:
            elapsed = time.time() - last_time_called[0]
            wait_time = max(0, (1 / calls_per_second) - elapsed)
            time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return result
    return wrapper

def retry_with_backoff(max_retries=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout) as e:
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"API call failed with error {e}. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                except Exception as e:
                    logging.error(f"Unhandled exception: {e}")
                    raise
            raise Exception("Max retries exceeded")
        return wrapper
    return decorator

# JSON Schemas for validation
code_analysis_schema = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string"},
        "key_functions": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "features": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "libraries": {
            "oneOf": [
                {"type": "object"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "usage_examples": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        }
    },
    "required": ["purpose", "key_functions", "features", "libraries", "usage_examples"]
}

# Trying a better way to force strict JSON, I will run it with gpt-4o-mini later to see if mini can cut it.
documentation_schema = {
    "type": "object",
    "properties": {
        "project_purpose": {"type": "string"},
        "project_functionalities": {"type": "array", "items": {"type": "string"}},
        "project_architecture": {"type": "object"},
        "project_notable_features": {"type": "array", "items": {"type": "string"}},
        "project_libraries": {"type": "object"},
        "project_examples": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["project_purpose", "project_functionalities", "project_architecture",
                 "project_notable_features", "project_libraries", "project_examples"]
}

# Configuration for the AutoGen Agents
config_list_openai = [
    {"model": model_name, "api_key": api_key}
]
llm_config = {
    "seed": semilla,
    "config_list": config_list_openai,
    "timeout": 60000,
    "temperature": temperature,
    "response_format": {'type': "json_object"},
}

# Initialize agents directly
def initialize_agents():
    """
    Initializes the agents required for code analysis and documentation.
    """
    code_analyzer = AssistantAgent(
        name="CodeAnalyzer",
        system_message="You are an expert at analyzing code. You provide detailed analysis on code snippets, focusing on purpose, key functions, features, libraries used, and practical usage examples. Always respond in JSON with keys 'purpose', 'key_functions', 'features', 'libraries', and 'usage_examples'.",
        llm_config=llm_config,
        max_consecutive_auto_reply=6,
        code_execution_config=False,
    )

    documentation_assistant = AssistantAgent(
        name="DocumentationAssistant",
        system_message="You are an expert in code analysis and documentation. Your task is to understand and document code from repositories. You provide comprehensive documentation including purpose, functionalities, architecture, notable features, and examples. Always respond in JSON with keys 'project_purpose', 'project_functionalities', 'project_architecture', 'project_notable_features', 'project_libraries', and 'project_examples'.",
        llm_config=llm_config,
        max_consecutive_auto_reply=6,
        code_execution_config=False,
    )

    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        llm_config=llm_config,
        system_message="You read and process code from repositories. You only respond in JSON."
    )

    return code_analyzer, documentation_assistant, user_proxy


class DependencyAnalyzer:
    """
    Analyzes dependencies in a repository.
    """

    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.dependencies = {
            "Python": [],
            # Extend with more languages as needed
        }

    def analyze_dependencies(self):
        """
        Analyzes dependencies by looking for specific files.
        """
        try:
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    if file == "requirements.txt":
                        self.dependencies["Python"].extend(self.parse_requirements_txt(file_path))
                    # Extend with more languages and files
        except Exception as e:
            logging.error(f"Error while analyzing dependencies: {e}")

        return self.dependencies

    def parse_requirements_txt(self, file_path):
        """
        Parses a Python requirements.txt file.
        """
        dependencies = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for req in pkg_resources.parse_requirements(f):
                    dependencies.append(str(req))
        except Exception as e:
            logging.error(f"Error parsing requirements.txt: {e}")
        return dependencies

class CodeAnalyzer:
    """
    Analyzes code files using the specified analyzer agent.
    """

    def __init__(self, analyzer_agent, user_proxy):
        self.analyzer_agent = analyzer_agent
        self.user_proxy = user_proxy

    @retry_with_backoff()
    @rate_limited_call
    def analyze_code_with_agent(self, file_path):
        """
        Analyzes a code file and returns the analysis result.
        """
        try:
            if not os.path.isfile(file_path):
                logging.error(f"File does not exist: {file_path}")
                return {"Error": "File not found"}

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                file_content = code_file.read()

            result = self.user_proxy.initiate_chat(
                self.analyzer_agent,
                silent=False,
                message = '''
# Analyze this code:
```{file_content}```

# Provide a comprehensive description of its purpose, key functions, notable features, libraries used, and usage examples.
Always respond in JSON with the following keys:
{
    "purpose": "string",
    "key_functions": ["array of strings"],
    "features": ["array of strings"],
    "libraries": ["array of strings"],
    "usage_examples": ["array of strings"]
}
Each item must be an individual string within an array. For example:
{
    "purpose": "To automate X process",
    "key_functions": ["Function A does X", "Function B handles Y"],
    "features": ["Feature 1", "Feature 2"],
    "libraries": ["library1", "library2"],
    "usage_examples": ["Example 1", "Example 2"]
}
'''
            )

            for message in reversed(result.chat_history):
                if message["name"] == self.analyzer_agent.name:
                    salida = self.safe_json_loads(message["content"])
                    if salida and self.validate_json_response(salida):
                        return salida
            return {"Error": "Analysis failed"}
        except Exception as e:
            logging.error(f"Unexpected error while analyzing file {file_path}: {e}")
            return {"Error": "Analysis failed"}

    def ensure_list(self, value):
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            return [value]
        else:
            return []

    # Modify `safe_json_loads` accordingly:
    def safe_json_loads(self, content):
        """
        Safely loads JSON content and ensures type conformance.
        """
        try:
            data = json.loads(content)
            # Ensure all fields have the correct types
            if 'key_functions' in data:
                data['key_functions'] = self.ensure_list(data['key_functions'])
            if 'features' in data:
                data['features'] = self.ensure_list(data['features'])
            if 'libraries' in data:
                data['libraries'] = self.ensure_list(data['libraries'])
            if 'usage_examples' in data:
                data['usage_examples'] = self.ensure_list(data['usage_examples'])
            if 'project_examples' in data:
                data['project_examples'] = self.ensure_list_of_strings(data['project_examples'])
            if 'project_libraries' in data and isinstance(data['project_libraries'], list):
                # Convert list to dictionary with placeholder keys
                data['project_libraries'] = {f"library_{i}": lib for i, lib in enumerate(data['project_libraries'])}
            return data
        except json.JSONDecodeError:
            logging.error("JSON decode error in code analysis message")
            return None


    def ensure_list_of_strings(self, value):
        if isinstance(value, list):
            return [self.convert_to_string(v) for v in value]
        elif isinstance(value, str):
            return [value]
        else:
            return []

    def convert_to_string(self, value):
        if isinstance(value, dict):
            # Convert complex dict to a readable string representation
            if 'description' in value and 'steps' in value:
                return f"{value['description']}. Steps: {'; '.join(value['steps'])}"
            return json.dumps(value, indent=2)
        elif isinstance(value, list):
            return ', '.join([str(v) for v in value])
        else:
            return str(value)

    def validate_json_response(self, response):
        """
        Validates the JSON response against the schema and logs detailed issues.
        """
        try:
            validate(instance=response, schema=code_analysis_schema)
            return True
        except ValidationError as e:
            logging.error(f"JSON validation error for field '{e.path}': {e.message}")
            return False

class RepositoryProcessor:
    """
    Processes the repository and generates documentation.
    """

    def __init__(self, repo_path, code_analyzer, documentation_assistant, user_proxy):
        self.repo_path = repo_path
        self.code_analyzer = code_analyzer
        self.documentation_assistant = documentation_assistant
        self.user_proxy = user_proxy
        self.summary = {}

    def repo_structure(self):
        """
        Analyzes the repository structure.
        """
        try:
            structure = ""
            for root, dirs, files in os.walk(self.repo_path):
                if '.git' in root or '.cache' in root:
                    continue
                level = root.replace(self.repo_path, '').count(os.sep)
                indent = ' ' * 4 * level
                structure += f'{indent}{os.path.basename(root)}/\n'
                sub_indent = ' ' * 4 * (level + 1)
                for file in files:
                    if file == '.DS_Store':
                        continue
                    structure += f'{sub_indent}{file}\n'
            return structure
        except Exception as e:
            logging.error(f"Error while analyzing repository structure: {e}")
            return ""

    def process_repository(self):
        """
        Main function to process the repository and generate documentation.
        """
        temp_structure = self.repo_structure()
        self.summary = {
            "project_name": os.path.basename(self.repo_path),
            "project_structure": temp_structure,
            "project_purpose": " ",
            "project_functionalities": [],
            "project_architecture": {},
            "project_notable_features": [],
            "project_libraries": {},
            "project_modules": {},
            "project_examples": []
        }

        # Analyze dependencies
        logging.info("Analyzing dependencies...")
        dependency_analyzer = DependencyAnalyzer(self.repo_path)
        dependencies = dependency_analyzer.analyze_dependencies()
        self.summary["project_libraries"] = dependencies

        # Collect all file paths to be analyzed
        file_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.repo_path)
            for file in files
            if file.endswith(('.py', '.js', '.java', '.cpp', '.h', '.txt', '.yml', '.yaml', '.ini', '.sh', '.env', '.json', '.md'))
        ]

        # Analyze files concurrently
        logging.info("Starting concurrent analysis of code files...")
        self.summary["project_modules"] = self.analyze_files_concurrently(file_paths)

        # Proceed to comprehensive documentation generation
        groupchat = GroupChat(
            agents=[self.documentation_assistant, self.user_proxy],
            messages=[],
            max_round=6,
            speaker_selection_method="auto",
            allow_repeat_speaker=False
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        conclusion = self.user_proxy.initiate_chat(
            manager,
            silent=False,
            code_execution_config=False,
            max_rounds=6,
            message=f'''
    # Here's the structure of the repository:
    {temp_structure}

    # Here is a Summary of the repository:
    {json.dumps(self.summary, indent=2)}

    # Please consolidate and provide comprehensive documentation including purpose, functionalities, architecture, notable features, libraries used, and usage examples. Format your answer in JSON with keys 'project_purpose', 'project_functionalities', 'project_architecture', 'project_notable_features', 'project_libraries', and 'project_examples'. TERMINATE when finished.'''
        )

        for message in reversed(conclusion.chat_history):
            if message["name"] == self.documentation_assistant.name:
                salida = self.safe_json_loads(message["content"])
                if salida and self.validate_json_response(salida):
                    self.summary.update(salida)
                    break
                else:
                    # Re-prompt the assistant to correct the JSON format
                    correction_result = self.user_proxy.initiate_chat(
                        self.documentation_assistant,
                        silent=False,
                        message="Please correct your previous response to be valid JSON according to the schema."
                    )

                    for message in reversed(correction_result.chat_history):
                        if message["name"] == self.documentation_assistant.name:
                            salida = self.safe_json_loads(message["content"])
                            if salida and self.validate_json_response(salida):
                                self.summary.update(salida)
                                break

        self.save_summary()


    def analyze_files_concurrently(self, file_paths):
        """
        Analyzes multiple files concurrently.
        """
        results = {}
        # Reminder max/workers comes from the YAML that I loaded at the beginning, minimum RPM must be 60, otherwise the int will be zero and will give an error
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.code_analyzer.analyze_code_with_agent, file_path): file_path for file_path in file_paths}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    results[module_name] = result
                except Exception as exc:
                    logging.error(f'Error while processing file {file_path}: {exc}')
        return results

    def safe_json_loads(self, content):
        """
        Safely loads JSON content.
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.error("JSON decode error in group chat message")
            return None

    def validate_json_response(self, response):
        """
        Validates the JSON response against the documentation schema.
        this is a placeholder for future validation, leave it as it is by now
        """
        try:
            validate(instance=response, schema=documentation_schema)
            return True
        except ValidationError as e:
            logging.error(f"JSON validation error: {e}")
            return False

    def save_summary(self):
        """
        Saves the summary in JSON and Markdown format.
        """
        try:
            with open('output.json', 'w') as json_file:
                json.dump(self.summary, json_file, indent=4)
            with open('output.md', 'w') as md_file:
                md_file.write(self.json_to_markdown(self.summary))
        except Exception as e:
            logging.error(f"Error while saving summary: {e}")

    def json_to_markdown(self, data, indent=0):
        """
        Helper function to convert JSON to Markdown.
        """
        if isinstance(data, dict):
            markdown = ""
            for key, value in data.items():
                markdown += f"{' ' * indent}## {key}\n\n"
                markdown += self.json_to_markdown(value, indent + 4)
            return markdown
        elif isinstance(data, list):
            markdown = ""
            for item in data:
                if isinstance(item, dict) or isinstance(item, list):
                    markdown += f"{' ' * indent}-\n{self.json_to_markdown(item, indent + 4)}"
                else:
                    markdown += f"{' ' * indent}- {item}\n"
            return markdown
        else:
            return f"{' ' * indent}{data}\n\n"

# Main function to execute the script
def main():
    """
    Main function to execute the script.
    """
    # Direct agent initialization
    code_analyzer_agent, documentation_assistant, user_proxy = initialize_agents()

    # Create instances of classes with these agents
    code_analyzer = CodeAnalyzer(code_analyzer_agent, user_proxy)
    repository_processor = RepositoryProcessor(repo_path, code_analyzer, documentation_assistant, user_proxy)

    # Process the repository
    repository_processor.process_repository()

if __name__ == "__main__":
    main()
