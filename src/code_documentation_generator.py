"""
Module: code_documentation_generator.py
Description:
    This script analyzes code repositories and generates comprehensive documentation
    including purpose, functionalities, architecture, notable features, libraries used,
    and usage examples. It utilizes OpenAI's language models via the AutoGen framework
    and includes enhanced error handling, concurrency management, and logging.

## Author Information
- **Author**: Nic Cravino
- **Email**: spidernic@me.com / ncravino@mac.com
- **LinkedIn**: [Nic Cravino](https://www.linkedin.com/in/nic-cravino)
- **Date**: October 26, 2024

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
        "key_functions": {"type": "array", "items": {"type": "string"}},
        "features": {"type": "array", "items": {"type": "string"}},
        "libraries": {"type": "array", "items": {"type": "string"}},
        "usage_examples": {"type": "array", "items": {"type": "string"}}
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

# New attempt to manage the agents differrently, don't like it much, will revert tot he array if I get anoyed.
class AgentManager:
    """
    Manages the initialization of AutoGen agents.
    """

    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.code_analyzer = None
        self.documentation_assistant = None
        self.user_proxy = None

    def initialize_agents(self):
        """
        Initializes the agents required for code analysis and documentation.
        """
        self.code_analyzer = AssistantAgent(
            name="CodeAnalyzer",
            system_message="You are an expert at analyzing code. You provide detailed analysis on code snippets, focusing on purpose, key functions, features, libraries used, and practical usage examples. Always respond in JSON with keys 'purpose', 'key_functions', 'features', 'libraries', and 'usage_examples'.",
            llm_config=self.llm_config,
            max_consecutive_auto_reply=6,
            code_execution_config=False,
        )

        self.documentation_assistant = AssistantAgent(
            name="DocumentationAssistant",
            system_message="You are an expert in code analysis and documentation. Your task is to understand and document code from repositories. You provide comprehensive documentation including purpose, functionalities, architecture, notable features, and examples. Always respond in JSON with keys 'project_purpose', 'project_functionalities', 'project_architecture', 'project_notable_features', 'project_libraries', and 'project_examples'.",
            llm_config=self.llm_config,
            max_consecutive_auto_reply=6,
            code_execution_config=False,
        )

        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            llm_config=self.llm_config,
            system_message="You read and process code from repositories. You only respond in JSON."
        )

        return self.code_analyzer, self.documentation_assistant, self.user_proxy

class DependencyAnalyzer:
    """
    Analyzes dependencies in a repository.
    """

    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.dependencies = {
            "Python": [],
            "Node.js": [],
            "Java": [],
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
                    elif file == "package.json":
                        self.dependencies["Node.js"].extend(self.parse_package_json(file_path))
                    elif file == "pom.xml":
                        self.dependencies["Java"].extend(self.parse_pom_xml(file_path))
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

    def parse_package_json(self, file_path):
        """
        Parses a Node.js package.json file.
        """
        dependencies = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
                for dep_type in ["dependencies", "devDependencies"]:
                    if dep_type in package_data:
                        dependencies.extend([f"{key}: {value}" for key, value in package_data[dep_type].items()])
        except Exception as e:
            logging.error(f"Error parsing package.json: {e}")
        return dependencies

    def parse_pom_xml(self, file_path):
        """
        Parses a Java pom.xml file.
        """
        dependencies = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            namespace = {'mvn': 'http://maven.apache.org/POM/4.0.0'}
            for dependency in root.findall(".//mvn:dependency", namespaces=namespace):
                group_id = dependency.find("mvn:groupId", namespaces=namespace).text
                artifact_id = dependency.find("mvn:artifactId", namespaces=namespace).text
                version_elem = dependency.find("mvn:version", namespaces=namespace)
                version = version_elem.text if version_elem is not None else 'No version specified'
                dependencies.append(f"{group_id}:{artifact_id}:{version}")
        except Exception as e:
            logging.error(f"Error parsing pom.xml: {e}")
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
                message=f'''
# Analyze this code:
```{file_content}```

# Provide a comprehensive description of its purpose, key functions, notable features, libraries used, and usage examples.
Format your answer in JSON with keys 'purpose', 'key_functions', 'features', 'libraries', and 'usage_examples'.'''
            )

            for message in reversed(result.chat_history):
                if message["name"] == self.analyzer_agent.name:
                    salida = self.safe_json_loads(message["content"])
                    if salida and self.validate_json_response(salida):
                        return salida
                    else:
                        # Re-prompt the assistant to correct the JSON format, experimental IRQ - I do not know if it can break a GroupChat ...testing
                        logging.warning("Reprompting, I hope it works ...")
                        correction_result = self.user_proxy.send_message(
                            self.analyzer_agent,
                            "Please correct your previous response to be valid JSON according to the schema."
                        )
                        # Check the corrected response
                        for message in reversed(correction_result.chat_history):
                            if message["name"] == self.analyzer_agent.name:
                                salida = self.safe_json_loads(message["content"])
                                if salida and self.validate_json_response(salida):
                                    return salida
            return {"Error": "Analysis failed"}
        except Exception as e:
            logging.error(f"Unexpected error while analyzing file {file_path}: {e}")
            return {"Error": "Analysis failed"}

    def safe_json_loads(self, content):
        """
        Safely loads JSON content.
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.error("JSON decode error in code analysis message")
            return None

    def validate_json_response(self, response):
        """
        Validates the JSON response against the schema.
        """
        try:
            validate(instance=response, schema=code_analysis_schema)
            return True
        except ValidationError as e:
            logging.error(f"JSON validation error: {e}")
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
        Note that I initilize the summary value-keys with blank/nill do the script respects ot fails on potential data-type errors. I know ...
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

        # Analyze files concurrently - I use futures, to lock a variable placeholder and wait.
        logging.info("Starting concurrent analysis of code files...")
        self.summary["project_modules"] = self.analyze_files_concurrently(file_paths)

        # Proceed to comprehensive documentation generation using Autogen group chat
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
                    correction_result = self.user_proxy.send_message(
                        self.documentation_assistant,
                        "Please correct your previous response to be valid JSON according to the schema."
                    )
                    # Check the corrected response
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

def main():
    """
    Main function to execute the script.
    """
    agent_manager = AgentManager(llm_config)

    # Trying a new way to init agents, not the usual array, lets see how it goes. I do not like it much, but hey, lets try it.
    code_analyzer_agent, documentation_assistant, user_proxy = agent_manager.initialize_agents()
    code_analyzer = CodeAnalyzer(code_analyzer_agent, user_proxy)
    repository_processor = RepositoryProcessor(repo_path, code_analyzer, documentation_assistant, user_proxy)
    repository_processor.process_repository()

if __name__ == "__main__":
    main()
