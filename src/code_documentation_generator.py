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
import time
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import yaml  # For configuration management
import openai
import shutil  # For removing directories
from importlib.metadata import version  # Replace pkg_resources
import xml.etree.ElementTree as ET
from jsonschema import validate, ValidationError
import random
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'debug.log')
        
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Clean up cache directory if it exists
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cache')
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        logging.info("Cleaned up cache directory")
    except Exception as e:
        logging.warning(f"Failed to clean up cache directory: {e}")

# Load environmental variables securely
load_dotenv(override=True)
api_key = os.getenv('OPEN_AI_API_KEY')
model_name = os.getenv('MODEL_NAME', 'gpt-4')

if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

if not api_key.startswith('sk-'):
    raise ValueError("Invalid OpenAI API key format. Must start with 'sk-'")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration variables
temperature = config.get('temperature', 0.2)
max_api_calls = config.get('max_api_calls_per_minute', 60)
seed = config.get('seed', 42)
repo_path = os.path.abspath(config.get('repo_path', "test_repo"))

# Configure OpenAI client
config_list_openai = [
    {
        "model": model_name,
        "api_key": api_key,
        "base_url": os.getenv('OPENAI_API_BASE', "https://api.openai.com/v1"),
        "price": [0.01, 0.03]  # Example pricing for prompt and completion tokens
    }
]

# Rate limiting mechanism
rate_lock = threading.Lock()
last_time_called = [0.0]

def rate_limited_call(func):
    def wrapper(*args, **kwargs):
        with rate_lock:
            elapsed = time.time() - last_time_called[0]
            wait_time = max(0, (1 / (max_api_calls / 60)) - elapsed)
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
def initialize_agents():
    """
    Initializes the agents required for code analysis and documentation.
    
    Returns:
        tuple: (code_analyzer, documentation_assistant, user_proxy) - The initialized agents
    """
    llm_config = {
        "config_list": config_list_openai,
        "temperature": temperature,
        "seed": seed
    }

    code_analyzer = AssistantAgent(
        name="code_analyzer",
        system_message="You are an expert at analyzing code. You provide detailed analysis on code snippets, focusing on purpose, key functions, features, libraries used, and practical usage examples. Always respond in JSON with keys 'purpose', 'key_functions', 'features', 'libraries', and 'usage_examples'.",
        llm_config=llm_config,
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )

    documentation_assistant = AssistantAgent(
        name="documentation_assistant",
        system_message="You are an expert in code analysis and documentation. Your task is to understand and document code from repositories. You provide comprehensive documentation including purpose, functionalities, architecture, notable features, and examples. Always respond in JSON with keys 'project_purpose', 'project_functionalities', 'project_architecture', 'project_notable_features', 'project_libraries', and 'project_examples'.",
        llm_config=llm_config,
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
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
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove any version specifiers
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        try:
                            pkg_version = version(package)
                            dependencies.append(f"{package}=={pkg_version}")
                        except Exception as e:
                            logging.warning(f"Could not get version for {package}: {e}")
                            dependencies.append(package)
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

            # Determine file type and set appropriate syntax highlighting
            file_extension = os.path.splitext(file_path)[1].lower()
            syntax_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                # Add more mappings as needed
            }
            syntax = syntax_map.get(file_extension, 'plaintext')

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                file_content = code_file.read()

            message_template = '''
# Analyze this code:
```{0}
{1}
```

# Provide a comprehensive description of its purpose, key functions, notable features, libraries used, and usage examples.
Always respond in JSON with the following keys:
{{
    "purpose": "string",
    "key_functions": ["array of strings"],
    "features": ["array of strings"],
    "libraries": ["array of strings"],
    "usage_examples": ["array of strings"]
}}
'''
            # Format the message using format() instead of f-string
            message = message_template.format(syntax, file_content)

            result = self.user_proxy.initiate_chat(
                self.analyzer_agent,
                silent=False,
                message=message
            )

            # Process the response
            for message in reversed(result.chat_history):
                if message.get("name") == self.analyzer_agent.name and message.get("content"):
                    try:
                        # Remove markdown formatting if present
                        content = message["content"]
                        if content.startswith("```json"):
                            content = content[7:-3].strip()
                        salida = self.safe_json_loads(content)
                        return salida
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error in code analysis message: {e}")
                        return {"Error": "Failed to parse analysis result"}
            
            return {"Error": "No valid response from analyzer"}

        except Exception as e:
            logging.error(f"Unexpected error while analyzing file {file_path}: {e}")
            return {"Error": str(e)}

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
        try:
            # Initialize summary structure
            self.summary = {
                "project_name": os.path.basename(self.repo_path),
                "project_structure": self.repo_structure(),
                "project_purpose": "",
                "project_functionalities": [],
                "project_architecture": {},
                "project_notable_features": [],
                "project_libraries": {},
                "project_modules": {},
                "project_examples": []
            }

            # Analyze dependencies
            dependency_analyzer = DependencyAnalyzer(self.repo_path)
            self.summary["project_libraries"] = dependency_analyzer.analyze_dependencies()

            # Get all code files
            code_files = []
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.java', '.cpp', '.c')):  # Add more extensions as needed
                        code_files.append(os.path.join(root, file))

            # Analyze code files
            if code_files:
                analysis_results = self.analyze_files_concurrently(code_files)
                for file_path, analysis in analysis_results.items():
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    if "Error" not in analysis:
                        self.summary["project_modules"][rel_path] = analysis
                    else:
                        logging.error(f"Analysis failed for {rel_path}: {analysis['Error']}")
                        self.summary["project_modules"][rel_path] = {"Error": "Analysis failed"}

            # Generate final documentation
            message = f'''
# Here's the structure of the repository:
{self.repo_structure()}

# Here is a Summary of the repository:
{json.dumps(self.summary, indent=2)}

# Please consolidate and provide comprehensive documentation including purpose, functionalities, architecture, notable features, libraries used, and usage examples. Format your answer in JSON with keys 'project_purpose', 'project_functionalities', 'project_architecture', 'project_notable_features', 'project_libraries', and 'project_examples'. TERMINATE when finished.
'''

            result = self.user_proxy.initiate_chat(
                self.documentation_assistant,
                message=message,
                silent=False
            )

            # Process the final documentation
            for message in reversed(result.chat_history):
                if message.get("name") == self.documentation_assistant.name and message.get("content"):
                    try:
                        content = message["content"]
                        if content.startswith("```json"):
                            content = content[7:-3].strip()
                        doc = self.safe_json_loads(content)
                        self.summary.update(doc)
                        break
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error in group chat message: {e}")

            # Save the documentation
            self.save_summary()
            return self.summary

        except Exception as e:
            logging.error(f"Error processing repository: {e}")
            return None

    def analyze_files_concurrently(self, file_paths):
        """
        Analyzes multiple files concurrently.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=int(max_api_calls / 60)) as executor:
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
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Save JSON summary
        json_path = os.path.join(reports_dir, 'output.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved JSON summary to {json_path}")

        # Save Markdown summary
        md_path = os.path.join(reports_dir, 'output.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.json_to_markdown(self.summary))
            logging.info(f"Saved Markdown summary to {md_path}")

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
