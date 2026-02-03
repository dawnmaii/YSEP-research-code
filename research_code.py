# -*- coding: utf-8 -*-
#Data preparation stage integrated

# @title 1. Install Python Packages (with Version Pin)
# Description: This cell installs all necessary Python libraries with a version pin for tree-sitter to ensure compatibility.
# It will then automatically restart the runtime.

print("⏳ Installing Python libraries with compatibility fix...")
# MODIFIED: Pinned tree-sitter to a known compatible version to fix the TypeError.
!pip install langchain langchain_community sentence-transformers faiss-cpu tiktoken GitPython pyyaml "tree-sitter~=0.20.4" tree-sitter-languages -q

import os

# @title 2. Configure, Define Functions, and Execute Pipeline (Re-aligned with Original Scripts)
# Description: This cell has been updated with the advanced file filtering and extraction logic from your original scripts.

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
import os
import shutil
import fnmatch
import hashlib
import json
import git
import yaml # For the YAML chunker
from typing import List, Dict, Any

from tree_sitter import Language, Parser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tree_sitter_languages import get_language, get_parser
print("✅ Imports loaded.")

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================
GITHUB_URL = "https://github.com/LauroSilveira/microservices-java-spring-boot.git"
LOCAL_REPO_DIR = "./microservices-java-spring-boot"
EXTRACTED_DIR = "./extracted_artifacts"
METADATA_FILE = "metadata.json"
OUTPUT_JSONL_FILE = "./structural_chunks.jsonl"
FAISS_INDEX_PATH = "./faiss_index_code_chunks"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
print("✅ Configuration defined.")

# ==============================================================================
# SECTION 3: FUNCTION DEFINITIONS (ALIGNED WITH ORIGINAL SCRIPTS)
# ==============================================================================

# --- NEW: Advanced Filtering Constants from original script ---
IGNORE_DIRS = {".git", ".github", ".vscode", ".idea", "node_modules", "vendor", "target", "build", "dist", "__pycache__", ".DS_Store", "test", "tests", "testdata"}
IGNORE_FILE_PATTERNS = [
    "*.pb.go", "*_pb2.py", "*_pb2_grpc.py", # Auto-generated protobuf code
    "*_test.go", "test_*.py", "*_test.py", "*.spec.js", "*.test.js", "*.test.cs", # Test files
    "*.css", "*.html", "*.htm", "*.svg", "*.png", "*.jpg", "*.jpeg", "*.gif", # Non-architectural assets
    "LICENSE", "*.pyc", "*.pyo", "*.lock", # Licenses, compiled files, and lock files
]
CONFIG_EXTENSIONS = (".yaml", ".yml", ".json", ".env", ".toml", ".ini", ".conf", ".tf", ".tfvars")
SOURCE_CODE_EXTENSIONS = (
    ".py", ".java", ".go", ".js", ".ts", ".tsx",  # Web/Backend
    ".cs", ".rs", ".cpp", ".cc", ".c", ".h",      # Systems/Enterprise
    ".rb", ".php", ".kt", ".swift",               # Mobile/Web
    ".proto", ".graphql", ".sql"                  # API/Data
)
KNOWN_CONFIG_FILENAMES = ["Dockerfile", "docker-compose.yml", "skaffold.yaml", "Chart.yaml", "values.yaml", "requirements.txt", "package.json", "pom.xml", "build.gradle", "go.mod", "Makefile", "cloudbuild.yaml"]
KEYWORDS_IN_PATH = ["k8s", "kubernetes", "charts", "helm", "manifests", "deploy", "config", "terraform", "infra", "cicd", "api", "spec", "proto"]


def is_binary(filepath):
    try:
        with open(filepath, 'rb') as f: return b'\x00' in f.read(1024)
    except IOError: return True

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""): sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# 2. Update Classification Logic
def classify_source_type(rel_path):
    path_lower = rel_path.lower().replace(os.sep, '/')
    filename_lower = os.path.basename(path_lower)
    ext_lower = os.path.splitext(filename_lower)[-1]

    # Configs
    if "dockerfile" in filename_lower: return "dockerfile"
    if "docker-compose" in filename_lower: return "docker-compose"
    if filename_lower in ["cargo.toml", "pom.xml", "build.gradle", "package.json", "go.mod", "requirements.txt"]: return "dependency-manifest"
    if ext_lower in (".yaml", ".yml"): return "kubernetes-manifest"

    # Languages
    ext_map = {
        ".go": "go", ".py": "python", ".java": "java",
        ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".cs": "c_sharp", ".rs": "rust",
        ".cpp": "cpp", ".cc": "cpp", ".c": "cpp",
        ".rb": "ruby", ".php": "php"
    }
    if ext_lower in ext_map: return f"{ext_map[ext_lower]}-source"

    return "unknown-artifact"

# NEW: is_relevant_file function from original script
def is_relevant_file(file_path):
    filename_lower = os.path.basename(file_path).lower()
    filepath_lower = file_path.lower()
    if filename_lower in (name.lower() for name in KNOWN_CONFIG_FILENAMES): return True
    if filepath_lower.endswith(CONFIG_EXTENSIONS): return True
    if filepath_lower.endswith(SOURCE_CODE_EXTENSIONS): return True
    if "readme" in filename_lower: return True
    if any(keyword.lower() in filepath_lower for keyword in KEYWORDS_IN_PATH): return True
    return False

# MODIFIED: A complete replacement of extract_artifacts with the original script's logic
def extract_artifacts(repo_url, local_repo_dir, extracted_dir, metadata_file):
    if not os.path.exists(local_repo_dir):
        print(f"📥 Cloning repository from {repo_url}...")
        git.Repo.clone_from(repo_url, local_repo_dir)
    else:
        print(f"📁 Repository already exists at {local_repo_dir}. Skipping clone.")

    if os.path.exists(extracted_dir): shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir, exist_ok=True)

    metadata = []
    repo_root_abs = os.path.abspath(local_repo_dir)
    print(f"\n🔎 Traversing repository and extracting architecturally significant files...")

    for root, dirs, files in os.walk(repo_root_abs, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS] # Filter directories
        for filename in files:
            # Filter files by pattern
            if any(fnmatch.fnmatch(filename, pattern) for pattern in IGNORE_FILE_PATTERNS):
                continue

            full_path = os.path.join(root, filename)
            if is_binary(full_path): continue

            # Check if the file is relevant
            if is_relevant_file(full_path):
                relative_path = os.path.relpath(full_path, repo_root_abs)
                target_path = os.path.join(extracted_dir, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(full_path, target_path)

                metadata.append({
                    "file_name": filename,
                    "relative_path": relative_path.replace(os.sep, '/'),
                    "sha256_hash": compute_sha256(full_path),
                    "source_type": classify_source_type(relative_path)
                })

    metadata_path = os.path.join(extracted_dir, metadata_file)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Extraction complete. {len(metadata)} files extracted. Metadata saved to {metadata_path}")


def setup_tree_sitter_realigned():
    lang_names = [
        "go", "python", "java", "javascript",
        "typescript", "rust", "c_sharp", "cpp"
    ]
    parsers = {lang: get_parser(lang) for lang in lang_names}
    languages = {lang: get_language(lang) for lang in lang_names}
    print(f"✅ Tree-sitter setup complete. Parsers and languages loaded for: {list(parsers.keys())}")
    return parsers, languages

# --- ADVANCED CHUNKING LOGIC (Unchanged from previous version) ---

def chunk_code_file(metadata, content, parsers, languages):
    source_type = metadata.get('source_type', '')

    # 1. Detect Language from Source Type (e.g. "rust-source" -> "rust")
    lang = None
    for supported_lang in languages.keys():
        if supported_lang in source_type:
            lang = supported_lang
            break

    # 2. Fallback to Generic if Language not loaded
    if not lang:
        yield from chunk_generic_text_file(metadata, content)
        return

    # 3. EXPANDED QUERIES (The "Secret Sauce")
    queries = {
        'go': """
            (function_declaration name: (identifier) @name) @chunk
            (method_declaration name: (field_identifier) @name) @chunk
            (type_declaration (type_spec name: (type_identifier) @name)) @chunk
        """,
        'python': """
            (function_definition name: (identifier) @name) @chunk
            (class_definition name: (identifier) @name) @chunk
        """,
        'java': """
            (class_declaration name: (identifier) @name) @chunk
            (interface_declaration name: (identifier) @name) @chunk
            (method_declaration name: (identifier) @name) @chunk
        """,
        'javascript': """
            (function_declaration name: (identifier) @name) @chunk
            (class_declaration name: (identifier) @name) @chunk
        """,
        'typescript': """
            (function_declaration name: (identifier) @name) @chunk
            (class_declaration name: (identifier) @name) @chunk
            (interface_declaration name: (type_identifier) @name) @chunk
        """,
        'rust': """
            (function_item name: (identifier) @name) @chunk
            (struct_item name: (type_identifier) @name) @chunk
            (impl_item type: (type_identifier) @name) @chunk
        """,
        'c_sharp': """
            (class_declaration name: (identifier) @name) @chunk
            (method_declaration name: (identifier) @name) @chunk
            (interface_declaration name: (identifier) @name) @chunk
        """,
        'cpp': """
            (function_definition declarator: (function_declarator declarator: (identifier) @name)) @chunk
            (class_specifier name: (type_identifier) @name) @chunk
        """
    }

    # 4. Parse
    try:
        tree = parsers[lang].parse(bytes(content, "utf8"))
        query = languages[lang].query(queries.get(lang, "")) # Graceful empty query if key missing
        captures = query.captures(tree.root_node)

        name_map = {node.parent.id: node.text.decode('utf8') for node, name in captures if name == "name"}
        chunk_index = 0

        # 5. Extract Chunks
        for node, capture_name in (c for c in captures if c[1] == "chunk"):
            unit_name = name_map.get(node.id, 'anonymous')
            unit_type = lang.capitalize() + " Block"

            # Simple header logic
            preamble = f"// Source: {metadata['relative_path']}\n// Language: {lang}\n// Item: {unit_name}\n\n"
            chunk_text = preamble + node.text.decode('utf8')

            output_meta = metadata.copy()
            output_meta['section_header'] = f"{unit_name} ({lang})"
            yield {"chunk_id": f"{metadata['sha256_hash']}_{chunk_index}", "chunk_text": chunk_text, **output_meta}
            chunk_index += 1

        # If no chunks found (e.g. empty file or weird syntax), fallback to generic
        if chunk_index == 0:
            yield from chunk_generic_text_file(metadata, content)

    except Exception as e:
        print(f"Error parsing {metadata['file_name']} as {lang}: {e}. Falling back to text chunking.")
        yield from chunk_generic_text_file(metadata, content)

def chunk_yaml_file(metadata, content):
    try:
        resources = list(yaml.safe_load_all(content))
        for i, resource in enumerate(resources):
            if not isinstance(resource, dict): continue
            kind = resource.get('kind', 'UnknownKind')
            name = resource.get('metadata', {}).get('name', 'unnamed')
            chunk_text_yaml = yaml.dump(resource, sort_keys=False)
            preamble = f"# Source File: {metadata['relative_path']}\n# This is a Kubernetes resource of kind '{kind}' named '{name}'.\n\n"
            output_meta = metadata.copy()
            output_meta['section_header'] = f"Kubernetes Resource: {kind}/{name}"
            yield {"chunk_id": f"{metadata['sha256_hash']}_{i}", "chunk_text": preamble + chunk_text_yaml, **output_meta}
    except Exception:
        yield from chunk_generic_text_file(metadata, content)

def chunk_generic_text_file(metadata, content):
    if content.strip():
        output_meta = metadata.copy()
        output_meta['section_header'] = f"Full Content: {metadata['file_name']}"
        yield {"chunk_id": f"{metadata['sha256_hash']}_0", "chunk_text": content, **output_meta}

def process_file(metadata, content, parsers, languages):
    source_type = metadata.get("source_type", "unknown-artifact")
    if source_type in ["kubernetes-manifest", "helm-config-file", "helm-template-file"]:
        yield from chunk_yaml_file(metadata, content)
    elif "source" in source_type:
        yield from chunk_code_file(metadata, content, parsers, languages)
    else:
        yield from chunk_generic_text_file(metadata, content)

def perform_chunking(input_dir, metadata_file, output_file, parsers, languages):
    with open(os.path.join(input_dir, metadata_file), 'r') as f: all_metadata = json.load(f)
    total_chunks = 0
    with open(output_file, 'w') as outfile:
        for i, item in enumerate(all_metadata):
            full_path = os.path.join(input_dir, item['relative_path'])
            if (i+1) % 50 == 0: print(f"  Processing file {i+1}/{len(all_metadata)}: {item['relative_path']}")
            if not os.path.exists(full_path): continue
            with open(full_path, 'r', errors='ignore') as content_file: content = content_file.read()
            for chunk in process_file(item, content, parsers, languages):
                outfile.write(json.dumps(chunk) + '\n'); total_chunks += 1
    print(f"✅ Chunking complete. Generated {total_chunks} chunks into {output_file}")

# --- DB Loading Function (Unchanged) ---
def create_and_save_vector_db(input_file, persist_dir, embedding_model_name):
    documents = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line); page_content = data.pop("chunk_text", "")
            documents.append(Document(page_content=page_content, metadata=data))
    print(f"Loaded {len(documents)} chunks from {input_file}.")
    if not documents: print("❌ No documents to process."); return
    print(f"Initializing embedding model: '{embedding_model_name}'...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    print(f"Building FAISS vector store from {len(documents)} chunks...");
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(persist_dir)
    print(f"✅ Vector store built and saved to '{persist_dir}'")
    loaded_vector_store = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    results = loaded_vector_store.similarity_search("test query", k=1)
    print(f"✅ Verification successful. Loaded store and performed a sample search.")

print("✅ All helper functions defined.")


# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================
print("\n--- Starting End-to-End Data Preparation Pipeline ---")
extract_artifacts(GITHUB_URL, LOCAL_REPO_DIR, EXTRACTED_DIR, METADATA_FILE)
parsers, languages = setup_tree_sitter_realigned()
perform_chunking(EXTRACTED_DIR, METADATA_FILE, OUTPUT_JSONL_FILE, parsers, languages)
create_and_save_vector_db(OUTPUT_JSONL_FILE, FAISS_INDEX_PATH, EMBEDDING_MODEL)
print("\n--- ✅ End-to-End Data Preparation Pipeline Finished Successfully! ---")







import os
import json
import time
from typing import List, Dict, Any, Optional
import uuid
import traceback
import re # For string cleaning

# Pip installs for notebook environments
# Ensure all packages are upgraded to compatible versions
# NOTE: langchain_legacy is included for potential backward compatibility,
# but LLMChain itself is being removed in favor of LCEL.
! pip install --upgrade langchain langchain_openai langchain_community langchain_core langchain_legacy neo4j tiktoken python-dotenv -q

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, __version__ as pydantic_version
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import PydanticOutputParser
# NOTE: LLMChain import has been removed.
# Later cells using LLMChain must be updated to use the LCEL pipe syntax:
# e.g., chain = prompt | llm | parser
import glob
import yaml
from dotenv import load_dotenv

# @title 2. Upgrade Core Libraries (Optional but Recommended)
! pip install --upgrade langchain langchain-openai openai "pydantic>=2.0.0,<3.0.0\" neo4j langchain_community tiktoken -q
print("Cell 2: Library upgrade attempt complete.")

# @title 3. Check Pydantic Version
import pydantic
pydantic_version = pydantic.__version__
print(f"Using Pydantic version: {pydantic_version}")
if not pydantic_version.startswith("2."):
    print("WARNING: Pydantic V2 is strongly recommended for PydanticOutputParser compatibility.")

# @title 3.1 Clone Microservices Java Spring Boot Repository (if not already present)
# Description: Clones the LauroSilveira/microservices-java-spring-boot repository to be analyzed.

import os

MICROSERVICES_REPO_URL = "https://github.com/LauroSilveira/microservices-java-spring-boot.git"
MICROSERVICES_REPO_PATH = "microservices-java-spring-boot" # Local directory name


if not os.path.exists(MICROSERVICES_REPO_PATH):
    print(f"Cloning repository '{MICROSERVICES_REPO_URL}' into '{MICROSERVICES_REPO_PATH}'...")
    # Use a simple git clone command
    clone_command_output = os.system(f"git clone {MICROSERVICES_REPO_URL} {MICROSERVICES_REPO_PATH}")
    if clone_command_output == 0:
        print("Repository cloned successfully.")
    else:
        print(f"ERROR: Failed to clone repository. Exit code: {clone_command_output}")
        print(f"Please ensure git is installed and the URL is accessible.")
else:
    print(f"Repository '{MICROSERVICES_REPO_PATH}' already exists. Skipping clone.")
    print("If you want a fresh clone, please delete the existing directory first.")

# Verify
if os.path.exists(MICROSERVICES_REPO_PATH) and os.listdir(MICROSERVICES_REPO_PATH):
    print(f"Successfully located repository at: {os.path.abspath(MICROSERVICES_REPO_PATH)}")
    print(f"Contents sample: {os.listdir(MICROSERVICES_REPO_PATH)[:5]}")
else:
    print(f"ERROR: Repository directory '{MICROSERVICES_REPO_PATH}' is empty or does not exist after clone attempt.")

CHUNKED_INPUT_FILE = "./structural_chunks.jsonl"
LLM_MODEL = "gpt-4o"
MAX_LLM_RETRIES = 2
LLM_RETRY_DELAY = 5

# ==============================================================================
# UPDATED: A more expressive, generalized schema for nodes and relationships
# ==============================================================================
ALLOWED_ENTITY_TYPES = {
    # Core compute units
    "Service", "Component",
    # API and Network related
    "APIGateway", "APIEndpoint", "Port",
    # Asynchronous communication
    "MessageBroker", "MessageTopic",
    # Data and Storage
    "Database",
    # Infrastructure and CI/CD
    "ConfigurationFile", "TechnologyStack", "ProgrammingLanguage", "Tool"
}

ALLOWED_PRIMARY_RELATIONSHIP_TYPES = {
    # Synchronous Communication
    "CONNECTS_TO", "EXPOSES_ENDPOINT", "EXPOSES_PORT",
    # Asynchronous Communication
    "PUBLISHES_TO", "SUBSCRIBES_TO",
    # Data Interaction
    "WRITES_TO", "READS_FROM",
    # General & Build-time
    "HAS_COMPONENT", "DEPENDS_ON", "WRITTEN_IN", "DEFINED_BY"
}
# ==============================================================================


# Global variables to be populated by service inference
INFERRED_TOP_LEVEL_SERVICES: List[str] = []
INFERRED_SERVICE_DIRECTORY_PATTERNS: Dict[str, List[str]] = {}

# --- Pydantic Version Check and Secrets Loading (remains the same) ---
import pydantic
pydantic_version = pydantic.__version__
print(f"Using Pydantic version: {pydantic_version}")
if not pydantic_version.startswith("2."):
    print("WARNING: Pydantic V2 is expected. Ensure correct version and restart if necessary.")

try:
    from google.colab import userdata
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    NEO4J_URI = userdata.get('NEO4J_URI')
    NEO4J_USERNAME = userdata.get('NEO4J_USERNAME')
    NEO4J_PASSWORD = userdata.get('NEO4J_PASSWORD')
    if all([OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        os.environ['NEO4J_URI'] = NEO4J_URI
        os.environ['NEO4J_USERNAME'] = NEO4J_USERNAME
        os.environ['NEO4J_PASSWORD'] = NEO4J_PASSWORD
        print("Successfully loaded secrets from Colab Secrets Manager.")
    else:
        raise ImportError
except ImportError:
    print("Attempting to load from .env file.")
    from dotenv import load_dotenv
    load_dotenv()

if not all(os.getenv(key) for key in ['OPENAI_API_KEY', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']):
    print("CRITICAL WARNING: One or more required secrets not found. Script may fail.")
else:
    print("Secrets/environment variables loaded successfully.")

print(f"\nUsing LLM Model: {LLM_MODEL}")
print(f"Input chunked file will be: {CHUNKED_INPUT_FILE}")

# @title 5. Define Knowledge Graph Schema (Updated for Traceability)
#              'evidence' field to capture the LLM's reasoning.

class KGEntity(BaseModel):
    """An entity in the knowledge graph, with evidence for its creation."""
    name: str = Field(description="The canonical, normalized name of the entity.")
    type: str = Field(description=f"The type of the entity. Must be one of: {', '.join(ALLOWED_ENTITY_TYPES)}")
    evidence: str = Field(description="A brief justification for why this entity was extracted, based on the source text.")


class KGTriplet(BaseModel):
    """A relationship with evidence for its creation."""
    subject: KGEntity
    relation: str = Field(description=f"Relationship type. Must be one of: {', '.join(ALLOWED_PRIMARY_RELATIONSHIP_TYPES)}")
    object: KGEntity
    evidence: str = Field(description="A brief justification for why this relationship was extracted, based on the source text.")
    # Optional attributes from your original schema are preserved
    protocol: Optional[str] = Field(None, description="Communication protocol (e.g., HTTP, gRPC, AMQP, TCP, JDBC, SMTP) if specified.")
    port: Optional[int] = Field(None, description="Target port number for this specific interaction, if specified.")
    http_method: Optional[str] = Field(None, description="HTTP method (e.g., GET, POST, PUT, DELETE) if applicable.")
    path_or_uri: Optional[str] = Field(None, description="Specific API path, URI, or resource identifier if relevant.")
    message_queue_name: Optional[str] = Field(None, description="Name of the message queue or topic, if applicable.")
    interaction_direction: Optional[str] = Field(None, description="Observed directionality (e.g., 'sends-to', 'receives-from').")
    interaction_details_text: Optional[str] = Field(None, description="Any other brief, relevant textual detail describing the interaction.")


class ExtractedKnowledge(BaseModel):
    """Structured output for all triplets extracted from a text chunk."""
    triplets: List[KGTriplet] = Field(description="A list of extracted triplets.", default_factory=list)


class IdentifiedServiceInfo(BaseModel):
    """Describes a single service instance identified from a configuration file or project structure."""
    name: str = Field(description="The canonical name of the service instance.")
    is_prebuilt: bool = Field(description="True if the service uses a pre-built image (e.g., from Docker Hub like redis, postgres) and likely has no local source code for the image itself. False if built from local source code.")
    source_directory_guess: Optional[str] = Field(None, description="The relative path to the service's source code directory if non-prebuilt and identifiable, e.g., './frontend'.")
    defining_file_path: str = Field(description="The path to the file where this service was primarily defined or identified.")
    evidence_or_reason: str = Field(description="Brief evidence or reason for identifying this as a service and its type (prebuilt/non-prebuilt).")


class DiscoveredServices(BaseModel):
    """Structured output for services discovered by analyzing project files."""
    services: List[IdentifiedServiceInfo] = Field(description="A list of discovered service instances.", default_factory=list)
    deployment_configs_analyzed: List[str] = Field(description="List of paths to deployment configuration files that were analyzed.", default_factory=list)


print("✅ Pydantic models updated for full traceability.")

# @title 3.5 Infer Top-Level Microservices (LLM-Based Analysis - REVISED)
# Description: Uses LLM to analyze repository structure and key files to infer top-level services.

import os
import glob
import yaml # For potentially pre-parsing some simple YAMLs if needed, or LLM can handle raw text
import time # Make sure time is imported for retries
from typing import List, Dict, Any, Optional # Ensure these are imported
# Ensure Pydantic models from Cell 5 (IdentifiedServiceInfo, DiscoveredServices) are available

def create_service_identification_prompt():
    """
    Creates the prompt template for the LLM to identify services from file content.
    Inspired by LLM4MDG's _identify_service_prompt.
    """
    # This parser will be for the output of THIS specific LLM call for service identification
    service_discovery_parser = PydanticOutputParser(pydantic_object=DiscoveredServices)

    # Define the list of file names that are strong indicators of service definitions
    # We will prioritize analyzing these.
    # This list should be refined based on the common patterns in your target repositories.
    # Example: prioritize root docker-compose, then k8s deployments, then individual Dockerfiles.

    # Updated prompt string based on the LLM4MDG example:
    identify_service_prompt_str = """As an expert in microservices architecture, you are tasked with analyzing the content of one or more provided project files from an open-source microservices-based project. Your objective is to identify each distinct service instance designed to run as part of this project based *only* on the provided file content and file paths.

# Task Instructions
- From the given file content(s) (e.g., docker-compose.yml, Kubernetes yaml file, Dockerfile), identify every service instance.
- For each service identified, you must provide:
    - `name`: A canonical, normalized identifier for the service instance (e.g., "frontend", "paymentservice").
    - `is_prebuilt`: A boolean. Set to `true` if the service uses a common pre-built public image (e.g., 'redis:latest', 'postgres:13', 'nginx') and there's no clear indication of local source code being built for *this specific service image*. Set to `false` if the service definition indicates it's built from local source code (e.g., a `build: ./path/to/source` in Docker Compose, or a Dockerfile in a local context that isn't just pulling a generic base image and running it).
    - `source_directory_guess`: If `is_prebuilt` is `false`, provide a relative path (e.g., "./frontend", "./src/paymentservice") to the likely source code directory for this service *if inferable from the context* (like a build context in a Dockerfile or Docker Compose). Otherwise, null.
    - `defining_file_path`: The path of the primary file from the input that led to this service's identification.
    - `evidence_or_reason`: A brief justification for identifying this as a service and for its prebuilt status (e.g., "Defined as a service in docker-compose.yml using image 'redis:alpine'", "Build context './frontend' implies local source code.").
- Focus on distinct, deployable service units. Do not identify general libraries or parent directories as services unless they are explicitly defined as deployable services in the provided file(s).
- If multiple files are provided, synthesize the information. Ensure service names are unique.

# Output Format
Your output should strictly obey the following JSON format, fitting the `DiscoveredServices` schema.
{format_instructions}

# Files to Analyze:
{files_content_and_paths}
"""

    prompt_template = ChatPromptTemplate.from_template(
        template=identify_service_prompt_str,
        partial_variables={"format_instructions": service_discovery_parser.get_format_instructions()}
    )
    return prompt_template, service_discovery_parser

def llm_identify_services_from_repo(
    repo_path: str,
    llm_client: Optional[ChatOpenAI],
    prompt_template: ChatPromptTemplate,
    parser: PydanticOutputParser
) -> List[IdentifiedServiceInfo]:
    """
    Identifies services by having an LLM analyze key files from the repository.
    The file count limit is removed, but content length limit remains critical.
    """
    if not llm_client:
        print("LLM client not initialized. Cannot perform LLM-based service identification.")
        return []

    key_file_patterns = [
        "**/deployment*.yaml", "**/deployment*.yml",
        "**/service*.yaml", "**/service*.yml",
        "**/Dockerfile","**/pom.xml",
    ]

    files_to_analyze_content_accumulator = [] # Use a different name to avoid confusion with the old limit

    # Adjusted based on typical context windows (e.g., gpt-4o is 128k tokens, gpt-3.5-turbo 4k/16k)
    # Content length is a rough proxy for token count. 1 token ~ 4 chars.
    # For a 16k token model, this is ~64k chars. For 128k, ~512k chars.
    # Start with a conservative limit, e.g., for a 16k model.
    max_total_content_length = 50000
    current_content_length = 0
    files_analyzed_paths = []

    print(f"Scanning for key service definition files in: {repo_path}")

    found_file_paths = []
    for pattern in key_file_patterns:
        if pattern.startswith("**/"):
            search_pattern = os.path.join(repo_path, "**", pattern.split("**/", 1)[1])
            found_file_paths.extend(glob.glob(search_pattern, recursive=True))
        else:
            found_file_paths.extend(glob.glob(os.path.join(repo_path, pattern), recursive=False))

    unique_file_paths = sorted(list(set(found_file_paths)), key=lambda p: (os.path.basename(p) != 'docker-compose.yml' and os.path.basename(p) != 'docker-compose.yaml', p))

    print(f"Found {len(unique_file_paths)} unique potential service definition files.")
    print(f"IMPORTANT: Attempting to process all found files in one LLM call, up to max_total_content_length ({max_total_content_length} characters).")
    print("If this exceeds your LLM's context window, it will fail. You may need to implement batching for very large repositories.")

    for file_path in unique_file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: # Added errors='ignore'
                content = f.read()

            # Check if adding this file would exceed the total content length limit
            if current_content_length + len(content) > max_total_content_length:
                print(f"INFO: Reached max_total_content_length ({max_total_content_length} chars). Analyzing collected files up to this point.")
                print(f"      Skipping remaining files, starting with: {file_path}")
                break

            relative_file_path = os.path.relpath(file_path, repo_path)
            files_to_analyze_content_accumulator.append(f"--- File: {relative_file_path} ---\n{content}\n\n")
            files_analyzed_paths.append(relative_file_path)
            current_content_length += len(content)

            # The 'if len(files_to_analyze_content) >= 10:' limit has been REMOVED as per your request.

        except Exception as e:
            print(f"Warning: Could not read or process file {file_path}: {e}")
            continue

    if not files_to_analyze_content_accumulator:
        print("No key service definition files found or readable in the repository path provided to analyze.")
        return []

    print(f"\nSending content from {len(files_to_analyze_content_accumulator)} file(s) (Total Chars: {current_content_length}) to LLM for service identification: {files_analyzed_paths[:5]}...")

    files_content_str = "".join(files_to_analyze_content_accumulator)
    llm_input = {"files_content_and_paths": files_content_str}
    identified_services_info: List[IdentifiedServiceInfo] = []
    llm_output_text = "" # Initialize for error printing

    for attempt in range(MAX_LLM_RETRIES):
        try:
            # Ensure prompt_template is correctly formatted and llm_client is the ChatOpenAI instance
            formatted_prompt = prompt_template.format_prompt(**llm_input).to_string()
            llm_response = llm_client.invoke(formatted_prompt)

            if hasattr(llm_response, 'content'):
                 llm_output_text = llm_response.content
            elif isinstance(llm_response, str):
                 llm_output_text = llm_response
            else:
                 llm_output_text = str(llm_response) # Fallback

            if not llm_output_text.strip(): # Check if output is just whitespace
                print(f"    Attempt {attempt + 1}/{MAX_LLM_RETRIES}: LLM returned empty or whitespace output for service ID.")
                if attempt == MAX_LLM_RETRIES - 1: raise Exception("LLM returned empty or whitespace output after max retries.")
                time.sleep(LLM_RETRY_DELAY * (attempt+1))
                continue

            parsed_output: DiscoveredServices = parser.parse(llm_output_text)
            if parsed_output and parsed_output.services:
                identified_services_info.extend(parsed_output.services)
                print(f"LLM successfully identified {len(parsed_output.services)} services in this batch.")
            else:
                print("LLM output parsed, but no services identified by the LLM in this batch.")
            break
        except Exception as e:
            print(f"    Error during LLM service identification (Attempt {attempt+1}/{MAX_LLM_RETRIES}): {type(e).__name__} - {e}")
            if attempt == MAX_LLM_RETRIES - 1:
                print(f"    Max retries reached for LLM service identification. Raw LLM output snippet: {llm_output_text[:500]}...")
                # traceback.print_exc() # Uncomment for full traceback during debugging
                return []
            time.sleep(LLM_RETRY_DELAY * (attempt+1))

    # Deduplicate services by name
    final_services_dict: Dict[str, IdentifiedServiceInfo] = {}
    for service_info in identified_services_info:
        norm_name_key = service_info.name.lower().replace("-","_").replace(" ","_").strip("._ /\\")
        if not norm_name_key : continue # Skip if name becomes empty after normalization

        if norm_name_key not in final_services_dict:
            final_services_dict[norm_name_key] = service_info
        else:
            # Basic merge: prefer non-prebuilt, or one with a source_dir if the other doesn't have it
            if not service_info.is_prebuilt and final_services_dict[norm_name_key].is_prebuilt:
                final_services_dict[norm_name_key] = service_info
            elif service_info.source_directory_guess and not final_services_dict[norm_name_key].source_directory_guess:
                 final_services_dict[norm_name_key] = service_info

    final_service_list = list(final_services_dict.values())
    print(f"Total unique services identified by LLM after deduplication: {len(final_service_list)}")
    return final_service_list

# (The create_service_identification_prompt() function and Pydantic models
#  IdentifiedServiceInfo, DiscoveredServices should be defined above this or in Cell 5 as previously discussed)
# Ensure llm (ChatOpenAI client) is initialized before this cell, or passed to the functions that use it.

print("LLM-based service identification function (llm_identify_services_from_repo) updated: file count limit removed.")

# @title 6. Helper Functions: Initialize LLM and Neo4j
# Description: Functions to set up the LLM client and the connection to your Neo4j database.

def initialize_llm(model_name: str = LLM_MODEL) -> Optional[ChatOpenAI]:
    """Initializes and returns the Langchain LLM client."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set/loaded.")
        return None
    try:
        llm = ChatOpenAI(model=model_name, temperature=0.0, request_timeout=120)
        print(f"LLM client initialized for model: {model_name}")
        return llm
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        return None

_NEO4J_GRAPH_INSTANCE = None

def initialize_neo4j_graph(force_reconnect=False) -> Optional[Neo4jGraph]:
    """Initializes and returns the Langchain Neo4jGraph connection. Caches connection."""
    global _NEO4J_GRAPH_INSTANCE
    if _NEO4J_GRAPH_INSTANCE is not None and not force_reconnect:
        print("Returning cached Neo4j connection.")
        return _NEO4J_GRAPH_INSTANCE

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        print("Error: NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD environment variables not set/loaded correctly.")
        return None
    try:
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password
        )
        graph.query("RETURN 1") # Test connection
        _NEO4J_GRAPH_INSTANCE = graph
        print(f"Successfully connected to Neo4j database.") # Removed URI for brevity/security
        return graph
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("Ensure Neo4j AuraDB is running, credentials are correct, and IP allow lists are configured if needed.")
        return None

# Initialize and test
llm = initialize_llm()
if llm:
    print("LLM ready.")
else:
    print("LLM initialization FAILED. Check API Key.")

neo4j_graph = initialize_neo4j_graph()
if neo4j_graph:
    print("Neo4j connection ready.")
else:
    print("Neo4j connection FAILED. Check credentials and DB status.")

# @title 3.7 Generate Documentation Outline using LLM & Convert to Markdown
# Description: Uses an LLM to propose a documentation outline based on repo structure
#              and identified services, then converts the outline to Markdown format.

import os
import json
import time # For retries
from typing import List, Dict, Any, Optional



# --- Pydantic Models for Documentation Outline (from previous response) ---
class DocSection(BaseModel):
    title: str
    subsections: Optional[List['DocSection']] = None
DocSection.model_rebuild() # For self-referencing

class DocOutline(BaseModel):
    title: str
    sections: List[DocSection]

# --- Helper function to get repository structure summary (from your script) ---
def get_repo_structure_summary(repo_path, max_depth=3, max_items_per_dir=5):
    # ... (Keep the full get_repo_structure_summary function definition from your script) ...
    summary = []
    exclude_dirs = {'.git', '.github', '.vscode', '.idea', 'node_modules', 'vendor', 'target', 'dist', 'build', '__pycache__'}
    exclude_files_general_ext = {'.lock', '.sum', '.pyc', '.gz', '.zip', '.jar', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.DS_Store'}
    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in exclude_dirs]
        filtered_files = [f for f in files if not f.startswith('.') and not any(f.lower().endswith(ext) for ext in exclude_files_general_ext)]
        files = filtered_files
        depth = root.replace(repo_path, '').count(os.sep)
        if depth > max_depth: dirs[:] = []; continue
        indent = "  " * depth
        dir_name = os.path.basename(root) if root != repo_path else '.'
        if dir_name == '.' and depth > 0 : dir_name = os.path.basename(os.path.dirname(root)) + "/" + dir_name if depth >0 else "."
        summary.append(f"{indent}{dir_name}/")
        items_to_show = dirs[:max_items_per_dir] + files[:max_items_per_dir]
        for item_name in items_to_show[:max_items_per_dir*2]: summary.append(f"{indent}  - {item_name}")
        if len(dirs) > max_items_per_dir or len(files) > max_items_per_dir: summary.append(f"{indent}  - ... (and more)")
        if len(summary) > 1000: summary.append("... (structure summary truncated)"); break
    return "\n".join(summary)


# --- Function to convert JSON outline to Markdown (from your script) ---
def convert_outline_to_markdown(outline_data: Dict[str, Any], initial_level: int = 1) -> str:
    # ... (Keep the full convert_outline_to_markdown function definition from your script) ...
    markdown_lines = []
    def format_title(title: str, level: int): return title
    def add_section_to_markdown(section_data: Dict[str, Any], level: int, numbering_prefix: str = ""):
        indent = "  " * (level - initial_level)
        title = format_title(section_data.get('title', 'Untitled Section'), level)
        markdown_lines.append(f"{indent}- {numbering_prefix}{title}")
        if "subsections" in section_data and section_data["subsections"]:
            for i, subsection in enumerate(section_data["subsections"]):
                sub_prefix = f"{numbering_prefix}{i+1}." if numbering_prefix else f"{i+1}."
                add_section_to_markdown(subsection, level + 1, sub_prefix + " ")
    doc_title_text = outline_data.get('title', 'Architecture Document')
    markdown_lines.append(f"# {doc_title_text}\n")
    if "sections" in outline_data:
        for i, section in enumerate(outline_data["sections"]):
            main_section_prefix = f"{i+1}."
            add_section_to_markdown(section, initial_level, main_section_prefix + " ")
    return "\n".join(markdown_lines)


# --- LLM Propose Documentation Outline function (REVISED PROMPT) ---
def llm_propose_documentation_outline(
    llm_client: Optional[ChatOpenAI],
    repo_path: str,
    inferred_services: List[str]
) -> Optional[Dict[str, Any]]:
    if not llm_client:
        print("LLM client not initialized. Cannot propose documentation outline.")
        return None

    print("Gathering repository information for documentation outline generation...")
    structure_summary = get_repo_structure_summary(repo_path)
    root_readme_path = os.path.join(repo_path, "README.md")
    root_readme_exists = os.path.exists(root_readme_path)
    docs_folder = os.path.join(repo_path, "docs")
    docs_folder_exists = os.path.exists(docs_folder)
    docs_folder_content_sample_paths = []
    if docs_folder_exists:
        try: docs_folder_content_sample_paths = [os.path.join("docs", item) for item in os.listdir(docs_folder)[:3]]
        except Exception: pass

    api_specs_paths = []
    # Example API spec collection (simplified)
    for root, _, files in os.walk(repo_path):
        if any(ignored in root for ignored in ['.git', 'node_modules', 'vendor', 'test']): continue
        for file in files:
            if ("openapi" in file.lower() or "swagger" in file.lower()) and file.lower().endswith((".yaml", ".yml", ".json")):
                api_specs_paths.append(os.path.relpath(os.path.join(root, file), repo_path))
            elif file.lower().endswith(".proto"):
                api_specs_paths.append(os.path.relpath(os.path.join(root, file), repo_path))
        if len(api_specs_paths) > 50: break

    repo_context_for_prompt = f"""
- Main Services Identified: {', '.join(inferred_services) if inferred_services else 'None explicitly listed'}
- Root README.md exists: {root_readme_exists}
- 'docs/' folder exists: {docs_folder_exists} (Sample content paths: {docs_folder_content_sample_paths if docs_folder_content_sample_paths else 'N/A'})
- Identified API Specifications (sample paths): {', '.join(api_specs_paths) if api_specs_paths else 'None explicitly found'}
- Key Directory Structure Highlights:
{structure_summary}
    """

    outline_parser = PydanticOutputParser(pydantic_object=DocOutline)

    prompt_text = f"""You are an expert technical writer tasked with creating an outline for a microservice architecture document.
Based on the provided repository summary, propose a logical documentation outline structured around common architectural views.

Repository Summary:
{repo_context_for_prompt}

Task:
Propose a comprehensive documentation outline. The main sections should reflect key architectural views.
The "Service Details" section should have a sub-section for each identified service.
Your proposed structure should include sections for:
1.  **Introduction:** High-level purpose of the system.
2.  **High-Level Architecture & Views:** A section for key diagrams or overviews.
3.  **Service Details:** A deep dive into each individual service.
4.  **API & Communication View:** How services communicate (API specs, async messaging).
5.  **Data Architecture & Persistence View:** How data is stored and managed.
6.  **Deployment & Infrastructure View:** How the system is built and deployed.
7.  **Cross-Cutting Concerns:** Topics like security, logging, monitoring.

Generate an outline that includes these main sections, and add relevant sub-sections based on the provided repository summary. For example, if API spec files were found, the "API & Communication View" section should have a subsection for API Reference. If no API specs were found, you can omit that specific subsection.

Output Format:
Your output MUST be a JSON object matching the `DocOutline` schema provided below.
{outline_parser.get_format_instructions()}

Proposed Documentation Outline (JSON):
"""
    print("\nSending request to LLM for documentation outline proposal (informed by architectural views)...")

    llm_output_text = ""
    # Ensure MAX_LLM_RETRIES and LLM_RETRY_DELAY are defined
    max_retries = MAX_LLM_RETRIES if 'MAX_LLM_RETRIES' in globals() else 2
    retry_delay = LLM_RETRY_DELAY if 'LLM_RETRY_DELAY' in globals() else 5

    for attempt in range(max_retries):
        try:
            response = llm_client.invoke(prompt_text)
            llm_output_text = response.content if hasattr(response, 'content') else str(response)
            if not llm_output_text.strip():
                if attempt == max_retries - 1: raise Exception("LLM returned empty output after max retries.")
                time.sleep(retry_delay * (attempt+1)); continue
            parsed_outline: DocOutline = outline_parser.parse(llm_output_text)
            print("LLM successfully proposed a documentation outline.")
            return parsed_outline.model_dump()
        except Exception as e:
            print(f"Error during LLM outline generation (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                print(f"Max retries reached for outline generation. LLM output: {llm_output_text[:500]}..."); return None
            time.sleep(retry_delay * (attempt+1))
    return None


# --- Execution part of the cell ---
if 'llm' in globals() and llm and \
   'MICROSERVICES_REPO_PATH' in globals() and os.path.exists(MICROSERVICES_REPO_PATH) and \
   'INFERRED_TOP_LEVEL_SERVICES' in globals():

    print("\n--- Generating Documentation Outline from Repository Structure (Informed by Architectural Views) ---")
    services_for_outline = INFERRED_TOP_LEVEL_SERVICES if INFERRED_TOP_LEVEL_SERVICES is not None else []

    documentation_outline_dict = llm_propose_documentation_outline(
        llm,
        MICROSERVICES_REPO_PATH,
        services_for_outline
    )

    if documentation_outline_dict:
        print("\nLLM Proposed Documentation Outline (JSON):")
        print(json.dumps(documentation_outline_dict, indent=2))

        print("\n--- Converting Outline to Markdown Format ---")
        markdown_outline = convert_outline_to_markdown(documentation_outline_dict)
        print("\nMarkdown Formatted Documentation Outline:")
        print("```markdown")
        print(markdown_outline)
        print("```")
    else:
        print("Failed to generate a documentation outline using the LLM.")
else:
    print("Prerequisites not met for LLM-based documentation outline generation.")
    if 'llm' not in globals() or not llm: print("  - LLM client not initialized (run Cell 6).")
    if 'MICROSERVICES_REPO_PATH' not in globals() or not os.path.exists(MICROSERVICES_REPO_PATH): print(f"  - Repository path '{MICROSERVICES_REPO_PATH if 'MICROSERVICES_REPO_PATH' in globals() else 'Not Set'}' not found (run Cell 3.1).")
    if 'INFERRED_TOP_LEVEL_SERVICES' not in globals(): print("  - INFERRED_TOP_LEVEL_SERVICES not populated (run Cell 3.6).")

# # @title 7. Helper Functions: Normalization (Revised for Component Logic & Relation Specificity)

!pip install wordsegment -q
print("Wordsegment library installed/verified.")

import re
import uuid
import logging
from typing import Optional, Dict, List, Any
import wordsegment

# --- 2. Initialize the wordsegment library ---
wordsegment.load()
print("Wordsegment dictionary loaded.")

# Get the logger for the current module
logger = logging.getLogger(__name__)

def get_parent_service_from_path_final(
    relative_path: str,
    service_dir_patterns: Dict[str, List[str]],
    top_level_services: List[str]
) -> Optional[str]:
    """
    Finds the parent service from a file path using a multi-stage strategy.
    """
    if not relative_path:
        return None

    normalized_path = relative_path.lower().replace("\\", "/")

    # Strategy 1: Check the pre-computed directory patterns (most specific match)
    if service_dir_patterns:
        for service_name, patterns in service_dir_patterns.items():
            for pattern in patterns:
                if normalized_path.startswith(pattern.lower().replace("\\", "/")):
                    return service_name

    # Strategy 2 (Fallback): If no pattern matches, check if the path starts
    # with a known service name (e.g., "order-management/...")
    for service_name in top_level_services:
        hyphenated_name = service_name.replace('_', '-')
        if normalized_path.startswith(f"{hyphenated_name}/"):
            return service_name
        no_separator_name = service_name.replace('_', '')
        if normalized_path.startswith(f"{no_separator_name}/"):
            return service_name

    # Strategy 3 (Advanced Fallback): Check if the path CONTAINS the service name as a full directory segment
    for service_name in top_level_services:
        hyphenated_name = service_name.replace('_', '-')
        if f"/{hyphenated_name}/" in normalized_path:
            return service_name

    return None

def normalize_entity_name(
    name: str,
    entity_type: str,
    chunk_metadata_context: Dict[str, Any],
    inferred_top_services: List[str],
    service_dir_patterns: Dict[str, List[str]],
    parent_service: Optional[str] = None
) -> str:
    """
    Normalizes any entity name with advanced logic for CamelCase and word segmentation.
    (Corrected to prevent double underscores).
    """
    original_name = name
    name_norm = name.strip()
    entity_type_lower = entity_type.strip().lower()

    if name_norm.lower() in ["the service", "this service", "service", "application", "the application", "the microservice"]:
        parent_from_path = get_parent_service_from_path_final(
            chunk_metadata_context.get("relative_path", ""), service_dir_patterns, inferred_top_services
        )
        if parent_from_path:
            logger.info(f"  [Normalize] Raw: '{original_name}' ({entity_type}) -> Contextual Parent: '{parent_from_path}'")
            return parent_from_path

    # --- CORRECTED CamelCase and Separator Handling ---
    # 1. Standardize existing separators to underscores
    name_norm = name_norm.replace("-", "_").replace(" ", "_").replace(".", "_")
    # 2. Insert underscores for CamelCase transitions
    name_norm = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', name_norm)
    name_norm = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name_norm)
    # 3. Convert to lowercase
    canonical_name = name_norm.lower()

    # --- Word Segmentation for concatenated strings ---
    if '_' not in canonical_name and len(canonical_name) > 12:
        segmented_words = wordsegment.segment(canonical_name)
        if len(segmented_words) > 1:
            canonical_name = '_'.join(segmented_words)
            logger.info(f"  [Normalize] Segmented '{original_name}' -> '{canonical_name}'")

    # Remove common suffixes
    suffixes_to_remove = ["_service", "_topic", "_queue", "_event", "_stream", "_application", "_app"]
    for suffix in suffixes_to_remove:
        if canonical_name.endswith(suffix):
            canonical_name = canonical_name[:-len(suffix)]

    # Handle plurals for message topics
    if entity_type_lower == "messagetopic" and canonical_name.endswith('s'):
        canonical_name = canonical_name[:-1]

    # Final cleanup of any invalid characters and extra underscores
    canonical_name = ''.join(char for char in canonical_name if char.isalnum() or char == '_')
    canonical_name = re.sub(r'_+', '_', canonical_name).strip('_') # Replaces multiple underscores with one

    # Apply component scoping
    if entity_type_lower == "component" and parent_service:
        scoped_name = f"{canonical_name}_{parent_service}"
        logger.info(f"  [Normalize] Raw: '{original_name}' ({entity_type}) -> Scoped to Parent: '{scoped_name}'")
        return scoped_name

    if not canonical_name:
        unnamed = f"unnamed_{entity_type_lower}"
        logger.info(f"  [Normalize] Raw: '{original_name}' ({entity_type}) -> Fallback: '{unnamed}'")
        return unnamed

    if original_name != canonical_name:
        logger.info(f"  [Normalize] Raw: '{original_name}' ({entity_type}) -> Final: '{canonical_name}'")

    return canonical_name

def normalize_relation_type(relation_str: str) -> str:
    """Normalizes the relationship type to a standard upper snake_case format."""
    rel_norm = relation_str.strip().lower()
    primary_relation_map = {
        "calls": "CONNECTS_TO", "invokes": "CONNECTS_TO", "connects to": "CONNECTS_TO",
        "publishes to": "PUBLISHES_TO", "sends message to": "PUBLISHES_TO",
        "subscribes to": "SUBSCRIBES_TO", "consumes from": "SUBSCRIBES_TO", "listens to": "SUBSCRIBES_TO",
        "writes to": "WRITES_TO", "reads from": "READS_FROM",
        "exposes port": "EXPOSES_PORT", "exposes endpoint": "EXPOSES_ENDPOINT",
        "depends on": "DEPENDS_ON", "defined by": "DEFINED_BY",
        "has component": "HAS_COMPONENT", "written in": "WRITTEN_IN",
    }
    for key, value in primary_relation_map.items():
        if key in rel_norm:
            return value
    normalized_relation = ''.join(char for char in rel_norm.upper().replace(" ", "_").replace("-", "_") if char.isalnum() or char == '_')
    return normalized_relation.strip("_") or "RELATED_TO"

print("✅ Advanced normalization helper functions are ready.")

global INFERRED_TOP_LEVEL_SERVICES, INFERRED_SERVICE_DIRECTORY_PATTERNS, IDENTIFIED_SERVICES_DETAILS

# Initialize them here before populating to ensure a clean run
INFERRED_TOP_LEVEL_SERVICES = []
INFERRED_SERVICE_DIRECTORY_PATTERNS = {}
IDENTIFIED_SERVICES_DETAILS: List[IdentifiedServiceInfo] = []

# Ensure 'llm' client and the prompt/parser functions from previous cells are available
service_id_prompt_template, service_id_parser = None, None
if 'llm' in globals() and llm:
    try:
        service_id_prompt_template, service_id_parser = create_service_identification_prompt()
        print("Service identification prompt and parser created for execution.")
    except Exception as e:
        print(f"Error creating service identification prompt/parser: {e}")
        service_id_prompt_template, service_id_parser = None, None
else:
    print("LLM client not available globally. LLM-based service identification cannot proceed.")

# Check if the repo path exists (cloned in an earlier step)
if 'MICROSERVICES_REPO_PATH' in globals() and os.path.exists(MICROSERVICES_REPO_PATH) and \
   llm and service_id_prompt_template and service_id_parser:

    print(f"\n--- Starting LLM-Based Service Identification from Repository: {MICROSERVICES_REPO_PATH} ---")
    IDENTIFIED_SERVICES_DETAILS = llm_identify_services_from_repo(
        MICROSERVICES_REPO_PATH, llm, service_id_prompt_template, service_id_parser
    )

    if IDENTIFIED_SERVICES_DETAILS:
        # Populate INFERRED_TOP_LEVEL_SERVICES with CANONICAL names by using the
        # exact same normalization function that the main pipeline uses.
        temp_inferred_services = set()
        for s_info in IDENTIFIED_SERVICES_DETAILS:
            # Use our "golden" normalization function from Cell 7.
            canonical_name = normalize_entity_name(
                name=s_info.name,
                entity_type="Service",
                chunk_metadata_context={},
                inferred_top_services=[],
                service_dir_patterns={}
            )

            if canonical_name and len(canonical_name) > 3 and not canonical_name.isdigit():
                 temp_inferred_services.add(canonical_name)

        INFERRED_TOP_LEVEL_SERVICES = sorted(list(temp_inferred_services))

        # Populate INFERRED_SERVICE_DIRECTORY_PATTERNS using the canonical names
        for service_info in IDENTIFIED_SERVICES_DETAILS:
            norm_service_name = normalize_entity_name(
                name=s_info.name, entity_type="Service", chunk_metadata_context={},
                inferred_top_services=[], service_dir_patterns={}
            )

            if norm_service_name in INFERRED_TOP_LEVEL_SERVICES:
                common_patterns = [
                    f"src/{norm_service_name}/",
                    f"cmd/{norm_service_name}/",
                    f"apps/{norm_service_name}/",
                    f"services/{norm_service_name}/",
                    f"{norm_service_name}/"
                ]
                if service_info.source_directory_guess:
                    clean_source_dir = service_info.source_directory_guess.strip("./\\").replace("\\", "/")
                    if clean_source_dir and clean_source_dir != norm_service_name:
                        INFERRED_SERVICE_DIRECTORY_PATTERNS[norm_service_name] = [f"{clean_source_dir}/"] + common_patterns
                    else:
                        INFERRED_SERVICE_DIRECTORY_PATTERNS[norm_service_name] = common_patterns
                else:
                     INFERRED_SERVICE_DIRECTORY_PATTERNS[norm_service_name] = common_patterns

        print(f"\n--- LLM-Based Service Identification Complete ---")
        print(f"Total unique services identified and detailed: {len(IDENTIFIED_SERVICES_DETAILS)}")
        print(f"Final CANONICAL INFERRED_TOP_LEVEL_SERVICES: {INFERRED_TOP_LEVEL_SERVICES}")
    else:
        print("LLM-based service identification did not yield any services, or prerequisites were missing.")
else:
    if 'MICROSERVICES_REPO_PATH' not in globals() or not os.path.exists(MICROSERVICES_REPO_PATH):
        print(f"ERROR: Repository path ('{MICROSERVICES_REPO_PATH if 'MICROSERVICES_REPO_PATH' in globals() else 'Not Set'}') not found. Run the clone cell.")
    else:
        print("LLM client or service identification prompt/parser not available. LLM-based service identification skipped.")
print(INFERRED_TOP_LEVEL_SERVICES, INFERRED_SERVICE_DIRECTORY_PATTERNS, IDENTIFIED_SERVICES_DETAILS)

# @title 8. Define ERE Chain for Source Code (Corrected Syntax)

ere_chain = None
ere_pydantic_parser = None

if llm:
    # This uses the traceable ExtractedKnowledge model from cell 5
    ere_pydantic_parser = PydanticOutputParser(pydantic_object=ExtractedKnowledge)

    entity_types_str = ", ".join(f'"{t}"' for t in sorted(list(ALLOWED_ENTITY_TYPES)))
    relationship_types_str = ", ".join(f'"{t}"' for t in sorted(list(ALLOWED_PRIMARY_RELATIONSHIP_TYPES)))

    # --- UPDATED GENERALIZED PROMPT ---
    code_prompt_template_str = f"""You are an expert system analyzing source code to build a knowledge graph. Your task is to identify architectural entities and their relationships by recognizing abstract patterns. For each element you extract, you MUST provide a brief `evidence` string.

CONTEXTUAL METADATA:
- Source File: '{{file_name}}' Source File Path: '{{relative_path}}'
- Section Header: '{{section_header}}'
- 'Inferred Top-Level Services': [{{inferred_top_level_services_list_str}}]

SCHEMA:
- Allowed Entity Types: [{entity_types_str}]
- Allowed Relationship Types: [{relationship_types_str}]

--- STANDARD OPERATING PROCEDURE (SOP) ---

**1. System-Level File Handling (SPECIAL CASE FIRST):**
   - **First, check if the `Source File Path` is a root-level configuration file (like `docker-compose.yml`, `pom.xml`).**
   - **IF IT IS:** Your goal is to identify ALL services and technologies defined within this file. **You MUST IGNORE the 'Infer Parent Service' rule.**
   - You MUST extract not only the 'Inferred Top-Level Services' but also any other defined infrastructure components (e.g., prometheus, kafka, zipkin) and classify them correctly (e.g., as `:TechnologyStack` or `:Tool`).
   - **Example for `docker-compose.yml`:**
     - **Input Code Snippet:**
       ```yaml
       services:
         order-service:
           image: jonas51/order-service:latest
           depends_on:
             - zipkin
             - prometheus
         prometheus:
           image: prom/prometheus:v2.37.1
       ```
     - **Correct Triplet Extraction:**
       - `(order-service:Service) -[:DEPENDS_ON]-> (zipkin:Tool)`
       - `(order-service:Service) -[:DEPENDS_ON]-> (prometheus:Tool)`

2.  **Infer Parent Service:** From the 'Source File Path', infer the parent `:Service` this code belongs to. The name of this service should align with one of the names from the 'Inferred Top-Level Services' list. This parent service is the subject of most `HAS_COMPONENT` and `DEPENDS_ON` relationships.

3.  **Entity Name Normalization (CRITICAL):** Before creating any entity, you MUST normalize its name to a canonical form.
    -   **For `:MessageTopic`:** Resolve placeholders like `${{{{...}}}}` and remove common suffixes like `topic`, `queue`, `event`, `s`, and separators like `-` or `_`.
        -   *Example 1:* `${{{{kafka.stock-changes-topic}}}}` becomes `stockchanges`.
        -   *Example 2:* `ordersTopic` becomes `orders`.
    -   **For `:Service`:** Use the canonical names from the 'Inferred Top-Level Services' list.

4.  **Relationship Extraction Rules:**
    -   **Service Discovery & Gateway Routing Pattern (CRITICAL):** When analyzing a service that acts as a client to a service discovery system (e.g., has `@EnableEurekaClient` annotation or a dependency on a discovery client library), you must look for its routing rules. If a routing rule uses a load-balanced URI (e.g., `lb://some-service`), you MUST infer a `CONNECTS_TO` relationship from the gateway to that service.
        -   **Example:**
            -   **Input Context:** A file `ApiGatewayApplication.java` contains `@EnableEurekaClient`. A separate file `application.properties` for the same service contains the line `spring.cloud.gateway.routes[0].uri=lb://order-service`.
            -   **Correct Triplet Extraction:**
                - `(order:Service) -[:DEPENDS_ON]-> (service-discovery:Service)`
                - `(api-gateway:Service) -[:DEPENDS_ON]-> (service-discovery:Service)`
                - `(api-gateway:Service) -[:CONNECTS_TO]-> (order:Service)`
    -   **Publishing pattern (`PUBLISHES_TO`):** Create this relationship when code actively **sends, publishes, or produces** a message. The evidence should point to the method call that performs the send.
    -   **Subscribing pattern (`SUBSCRIBES_TO`):** Create this relationship ONLY when you see code that **actively listens for messages**.
        -   **Strong Evidence :** An **annotation or decorator** on a message-handling function (e.g., `@KafkaListener`, `@RabbitListener`, `@Subscribe`) or a **callback function** being registered with a client (e.g., `client.subscribe("topic", message_handler)`).
        -   **Weak Evidence (DO NOT USE for SUBSCRIBES_TO):** A function that only **creates and returns** a `Factory`, `Builder`, `Template`, or `Connection` object is just configuration. Do not infer a subscription from this alone.
    -   **Direct Connection Pattern:** For direct synchronous calls (e.g., `@FeignClient`), use `CONNECTS_TO`.

5.  **Directional Rule Example (CRITICAL):** You MUST determine the correct direction for connections. Follow this example carefully:
    -   **Input Code (from a file like `adservice.yaml`):**
        ```yaml
        ingress:
        - from:
          - podSelector:
              matchLabels:
                app: frontend
        ```
    -   **Correct Interpretation:** The keyword "ingress" means traffic is coming **IN TO** the service being configured (adservice) **FROM** the `frontend`. Therefore, the `frontend` is the source of the connection.
    -   **Correct Triplet:** `(frontend) -[:CONNECTS_TO]-> (adservice)`

6.  **Evidence Requirement:** For every single entity and triplet you extract, you MUST provide a concise `evidence` string.

--- GUIDING PRINCIPLES OF EVIDENCE ---

1.  **Prioritize Production Code:** Give the highest weight to evidence from application source code (e.g., `.java`, `.go`, `.py`) and declarative configurations (e.g., Kubernetes `.yaml`, Terraform `.tf`). These files define the actual architecture.

2.  **Be Skeptical of Non-Production Code:** Treat the following file types as having lower authority. The relationships described within them may be indirect, simulated, or purely descriptive.

    -   **Test Files (`/test/`, `*_test.go`):** Connections in test files are often to mock objects, not real services. Do not extract relationships from tests unless they clearly describe an integration test between two real services.
    -   **Load Generator Files (`locustfile.py`):** These files simulate user behavior. The HTTP calls made from these files are almost always directed at a single entry point (like a `frontend` or API gateway), not directly to backend services. **Do not infer direct service-to-service connections from these files.**
    -   **Documentation (`.md`):** These files contain descriptions of the architecture, which can be abstract or outdated. Use them to understand the system, but prefer evidence from source code when there is a conflict.

IMPORTANT: Your output MUST be a valid JSON object matching the `ExtractedKnowledge` schema.
{{format_instructions}}

Code to Analyze:
```
{{chunk_text_input}}
```
"""
    code_prompt = ChatPromptTemplate.from_template(
        template=code_prompt_template_str,
        partial_variables={"format_instructions": ere_pydantic_parser.get_format_instructions()}
    )

    ere_chain = code_prompt | llm | ere_pydantic_parser
    print("✅ Generalized and Traceable ERE Chain created successfully.")
else:
    print("❌ LLM not initialized. Cannot create ERE chain.")

import logging
import os
import json
import time
import uuid

# --- SCRIPT CONFIGURATION ---
RUN_FULL_PROCESSING = True
CLEAR_DATABASE_FIRST = True
LOG_LEVEL = logging.INFO

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("Logger for the final processing loop initialized.")

# --- MAIN EXECUTION BLOCK ---
if RUN_FULL_PROCESSING:
    logger.info("Starting final processing loop with direct storage and full validation.")

    prerequisites_ok = all([
        'ere_chain' in globals() and ere_chain,
        'ere_pydantic_parser' in globals() and ere_pydantic_parser,
        'neo4j_graph' in globals() and neo4j_graph,
        'INFERRED_TOP_LEVEL_SERVICES' in globals() and INFERRED_TOP_LEVEL_SERVICES,
        'ALLOWED_ENTITY_TYPES' in globals(),
        'ALLOWED_PRIMARY_RELATIONSHIP_TYPES' in globals()
    ])

    if not prerequisites_ok:
        logger.error("❌ Prerequisites not met. Please run all preceding cells successfully.")
    else:
        logger.info("✅ Prerequisites are met.")
        all_chunks_data = []
        if os.path.exists(CHUNKED_INPUT_FILE):
            with open(CHUNKED_INPUT_FILE, 'r', encoding='utf-8') as infile:
                all_chunks_data = [json.loads(line) for line in infile]
            logger.info(f"✅ Loaded {len(all_chunks_data)} code chunks.")
        else:
            logger.error(f"❌ ERROR: Input file '{CHUNKED_INPUT_FILE}' not found.")

        if all_chunks_data and CLEAR_DATABASE_FIRST:
            logger.info("\n--- Clearing Neo4j Database ---")
            neo4j_graph.query("MATCH (n) DETACH DELETE n")
            logger.info("✅ Neo4j database cleared successfully.")

        if all_chunks_data:
            logger.info(f"\n--- STARTING KNOWLEDGE GRAPH EXTRACTION & DIRECT LOADING ({len(all_chunks_data)} chunks) ---")

            total_chunks_processed = 0
            total_triplets_extracted = 0
            total_triplets_stored = 0
            inferred_services_str_for_prompt = ", ".join(INFERRED_TOP_LEVEL_SERVICES)
            logger.info(f"Using top-level services for context: {inferred_services_str_for_prompt}")

            for i, chunk_data in enumerate(all_chunks_data):
                total_chunks_processed += 1
                chunk_id = chunk_data.get("chunk_id", f"chunk_{i}")
                chunk_text = chunk_data.get("chunk_text")

                if (i % 25 == 0):
                    logger.info(f"\n--- Processing chunk {i+1}/{len(all_chunks_data)} ---")
                if not chunk_text: continue

                prompt_input_dict = {
                    "file_name": chunk_data.get("file_name", "N/A"),
                    "relative_path": chunk_data.get("relative_path", "N/A"),
                    "section_header": chunk_data.get("section_header", "N/A"),
                    "chunk_text_input": chunk_text,
                    "inferred_top_level_services_list_str": inferred_services_str_for_prompt
                }

                extracted_knowledge_obj = None
                for attempt in range(MAX_LLM_RETRIES):
                    try:
                        # Invoke the chain
                        response = ere_chain.invoke(prompt_input_dict)

                        # Case A: Chain returns the Pydantic object directly (The cause of your error)
                        if hasattr(response, 'triplets'):
                            extracted_knowledge_obj = response

                        # Case B: Chain returns a dictionary (Legacy LangChain behavior)
                        elif isinstance(response, dict):
                            llm_text_output = response.get('text', '')
                            if llm_text_output:
                                extracted_knowledge_obj = ere_pydantic_parser.parse(llm_text_output)

                        # Case C: Chain returns a raw string
                        elif isinstance(response, str):
                            extracted_knowledge_obj = ere_pydantic_parser.parse(response)

                        break # Success, exit retry loop
                    except Exception as e_llm:
                        if attempt == MAX_LLM_RETRIES - 1: logger.error(f"LLM/Parser error on final attempt for chunk {chunk_id}: {e_llm}")
                        time.sleep(LLM_RETRY_DELAY)

                if not extracted_knowledge_obj or not extracted_knowledge_obj.triplets: continue
                total_triplets_extracted += len(extracted_knowledge_obj.triplets)

                for raw_triplet in extracted_knowledge_obj.triplets:
                    logger.info("-" * 60)
                    logger.info(f"Processing Raw Triplet: ('{raw_triplet.subject.name}', '{raw_triplet.relation}', '{raw_triplet.object.name}')")

                    s_original = raw_triplet.subject.model_copy(deep=True)
                    o_original = raw_triplet.object.model_copy(deep=True)
                    s, o, rel = raw_triplet.subject, raw_triplet.object, normalize_relation_type(raw_triplet.relation)

                    parent_service = get_parent_service_from_path_final(
                        chunk_data["relative_path"],
                        INFERRED_SERVICE_DIRECTORY_PATTERNS,
                        INFERRED_TOP_LEVEL_SERVICES
                    )
                    logger.info(f"  [Context] File Path: '{chunk_data.get('relative_path', 'N/A')}' -> Determined Parent Service: '{parent_service}'")

                    s.name = normalize_entity_name(s.name, s.type, chunk_data, INFERRED_TOP_LEVEL_SERVICES, INFERRED_SERVICE_DIRECTORY_PATTERNS, parent_service=parent_service)
                    o.name = normalize_entity_name(o.name, o.type, chunk_data, INFERRED_TOP_LEVEL_SERVICES, INFERRED_SERVICE_DIRECTORY_PATTERNS, parent_service=parent_service)

                    is_valid = True
                    generated_this_triplet = []
                    for entity, original_entity in [(s, s_original), (o, o_original)]:
                        if original_entity.type == "Service" and entity.name not in INFERRED_TOP_LEVEL_SERVICES:
                            if parent_service and parent_service in INFERRED_TOP_LEVEL_SERVICES:
                                logger.info(f"  [Validation] Demoting '{original_entity.name}' to a Component of '{parent_service}'.")
                                entity.type = "Component"
                                entity.name = normalize_entity_name(original_entity.name, "Component", chunk_data, INFERRED_TOP_LEVEL_SERVICES, INFERRED_SERVICE_DIRECTORY_PATTERNS, parent_service=parent_service)

                                generated_this_triplet.append(KGTriplet(
                                    subject=KGEntity(name=parent_service, type="Service", evidence="Inferred parent from file path."),
                                    relation="HAS_COMPONENT", object=entity,
                                    evidence=f"Inferred because '{original_entity.name}' was a non-top-level service."
                                ))
                            else:
                                logger.warning(f"  [Validation] Skipping rogue service '{original_entity.name}' with no valid parent.")
                                is_valid = False; break

                    if not is_valid:
                        logger.info("-" * 60 + "\n")
                        continue

                    logger.info(f"  [Validation] Triplet passed validation.")

                    triplets_to_store_now = [raw_triplet] + generated_this_triplet
                    raw_triplet.subject, raw_triplet.object, raw_triplet.relation = s, o, rel

                    for triplet in triplets_to_store_now:
                        try:
                            logger.info(f"  [Storage] Storing -> ({triplet.subject.name}:{triplet.subject.type}) -[:{triplet.relation}]-> ({triplet.object.name}:{triplet.object.type})")

                            for entity in [triplet.subject, triplet.object]:
                                # NEW QUERY (with n.path)
                                neo4j_graph.query(
                                    f"MERGE (n:`{entity.type}` {{name: $name}}) ON CREATE SET n.source_chunk_id = $chunk_id, n.source_file = $file, n.evidence = $evidence, n.path = $file",
                                    params={"name": entity.name, "chunk_id": chunk_id, "file": chunk_data.get("relative_path"), "evidence": entity.evidence}
                                )
                            rel_props = {"source_chunk_id": chunk_id, "evidence": triplet.evidence}
                            if hasattr(triplet, 'message_queue_name') and triplet.message_queue_name:
                                rel_props['message_queue_name'] = triplet.message_queue_name
                            neo4j_graph.query(
                                f"MATCH (s:`{triplet.subject.type}` {{name: $s_name}}), (o:`{triplet.object.type}` {{name: $o_name}}) MERGE (s)-[r:`{triplet.relation}`]->(o) SET r += $props",
                                params={"s_name": triplet.subject.name, "o_name": triplet.object.name, "props": rel_props}
                            )
                            total_triplets_stored += 1
                        except Exception as e_neo4j:
                            logger.error(f"  Neo4j storage error for triplet ({triplet.subject.name}, {triplet.relation}, {triplet.object.name}): {e_neo4j}")

                    logger.info("-" * 60 + "\n")

            logger.info("\n--- KNOWLEDGE GRAPH EXTRACTION COMPLETE ---")
            logger.info(f"Total chunks processed: {total_chunks_processed}")
            logger.info(f"Total triplets extracted by LLM: {total_triplets_extracted}")
            logger.info(f"Total triplets successfully stored in Neo4j: {total_triplets_stored}")
else:
    logger.info("RUN_FULL_PROCESSING is set to False. Skipping execution.")

# @title New Cell: List All Relationships in Knowledge Graph
# Description: Queries Neo4j to list all relationships, showing source node,
#              relationship type, relationship properties, and target node.

# Ensure Optional, List, Dict, Any are imported if this cell is run standalone
from typing import List, Dict, Any, Optional

def list_all_relationships_in_graph(graph_db: Optional[Neo4jGraph], limit: int = 1000):
    """
    Queries and prints a sample of all relationships from the Neo4j graph.
    """
    if not graph_db:
        print("Neo4j graph connection not available. Cannot list relationships.")
        return

    print(f"\n--- Listing All Relationships from Neo4j Graph (Limited to {limit} Results) ---")

    # Cypher query to find all relationships
    # Fetches source node name & label, relationship type & all its properties, target node name & label
    query = """
    MATCH (source_node)-[rel]->(target_node)
    RETURN
        source_node.name AS source_name,
        labels(source_node)[0] AS source_label,
        type(rel) AS relationship_type,
        properties(rel) AS relationship_properties,
        target_node.name AS target_name,
        labels(target_node)[0] AS target_label
    LIMIT $limit_val
    """

    try:
        results = graph_db.query(query, params={"limit_val": limit})

        if not results:
            print("No relationships found in the graph.")
        else:
            print(f"Found {len(results)} relationships (displaying up to {limit}):")
            for i, record in enumerate(results):
                source_name = record.get('source_name', 'UnknownSource')
                source_label = record.get('source_label', 'UnknownLabel')
                rel_type = record.get('relationship_type', 'UNKNOWN_REL')
                rel_props = record.get('relationship_properties', {})
                target_name = record.get('target_name', 'UnknownTarget')
                target_label = record.get('target_label', 'UnknownLabel')

                # Format properties for printing, excluding potentially long or internal ones if desired
                props_str_parts = []
                for k, v in rel_props.items():
                    if k not in ["source_chunk_id", "source_file"]: # Example: exclude some common internal props
                        props_str_parts.append(f"{k}: '{str(v)[:50]}'") # Truncate long prop values
                props_display = ", ".join(props_str_parts)
                if props_display:
                    props_display = f" {{{props_display}}}"
                else:
                    props_display = ""

                print(f"  {i+1}. ({source_label}:{source_name}) "
                      f"-[:{rel_type}{props_display}]-> "
                      f"({target_label}:{target_name})")

        # Optional: Get a count of all distinct relationship types present
        distinct_rel_types_query = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type, count(r) AS count ORDER BY count DESC"
        distinct_rel_types_results = graph_db.query(distinct_rel_types_query)
        if distinct_rel_types_results:
            print("\n--- Distinct Relationship Types and Counts ---")
            for rec in distinct_rel_types_results:
                print(f"  - Type: {rec['rel_type']}, Count: {rec['count']}")

    except Exception as e:
        print(f"Error querying for relationships: {e}")
        traceback.print_exc()

# --- Execution part of the cell ---
# This cell assumes 'neo4j_graph' (Cell 6) is initialized.

if 'neo4j_graph' in globals() and neo4j_graph:
    # You can adjust the limit for how many relationships to display
    list_all_relationships_in_graph(neo4j_graph, limit=1000)
else:
    print("Neo4j graph not initialized. Please run Cell 6 to initialize the connection.")

# @title Visualize Aggregated Relationships Between INFERRED Top-Level Services (Corrected)
# Description: Fetches and visualizes an AGGREGATED view of relationships
#              between the services identified by the inference functions.

# Ensure pyvis is available
!pip install pyvis -q
from pyvis.network import Network
from IPython.core.display import display, HTML
PYVIS_AVAILABLE = True

from typing import List, Dict, Any, Optional
import json

def visualize_aggregated_service_dependencies(
    graph_db: Optional[Neo4jGraph],
    inferred_services: List[str],
    max_rels_to_visualize: int = 500
):
    """
    Visualizes an aggregated view of relationships exclusively between the inferred service names.
    This version groups multiple relationships between two services into a single edge for clarity.
    """
    if not graph_db:
        print("Neo4j graph connection not available.")
        return
    if not PYVIS_AVAILABLE:
        print("pyvis library not available. Cannot visualize.")
        return
    if not inferred_services:
        print("The list of inferred services is empty. Nothing to visualize.")
        return

    target_service_names = inferred_services

    print(f"\n--- Visualizing Aggregated Relationships Between {len(target_service_names)} Inferred Services ---")

    # --- MODIFIED AGGREGATION QUERY ---
    # This query finds all directed relationships between the specified services.
    # It then groups them by the source (s1) and target (s2) nodes.
    # `collect(type(r))` gathers all relationship types (e.g., CONNECTS_TO, DEPENDS_ON) into a list.
    # `collect(properties(r))` gathers all properties from those relationships into a list of maps.
    viz_query = """
    MATCH (s1:Service)-[r]->(s2:Service)
    WHERE s1.name IN $service_names AND s2.name IN $service_names
    WITH s1, s2,
         collect(type(r)) AS rel_types,
         collect(properties(r)) AS rel_props_list
    RETURN
        s1.name AS source_name,
        s2.name AS target_name,
        rel_types,
        rel_props_list
    LIMIT $limit
    """
    query_params = {
        "service_names": target_service_names,
        "limit": max_rels_to_visualize
    }

    try:
        results = graph_db.query(viz_query, params=query_params)

        if not results:
            print("No direct relationships found between the inferred top-level service nodes to visualize.")
            return

        net = Network(notebook=True, cdn_resources='remote', height="800px", width="100%", directed=True, select_menu=True, filter_menu=True)
        net.set_options("""
        var options = {
          "nodes": { "font": { "size": 16, "face": "Tahoma" }, "shape": "box", "margin": 10, "scaling": {"min": 25, "max": 45}},
          "edges": {
            "font": { "size": 10, "align": "top", "color": "#404040", "strokeWidth": 0 },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 }},
            "smooth": { "type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.4 }
          },
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -100, "centralGravity": 0.02, "springLength": 200, "springConstant": 0.1, "damping": 0.4, "avoidOverlap": 0.5},
            "stabilization": {"iterations": 250}
          },
          "interaction": {"hover": true, "tooltipDelay": 200, "navigationButtons": true, "keyboard": true, "multiselect": true },
          "manipulation": { "enabled": true }
        }
        """)

        added_nodes = set()
        service_color = "#FF6347"  # Tomato

        for record in results:
            source_name = record["source_name"]
            target_name = record["target_name"]

            # Add nodes to the graph if they haven't been added yet
            for node_name in [source_name, target_name]:
                if node_name not in added_nodes:
                    title = f"Service: {node_name}"
                    net.add_node(node_name, label=node_name, title=title, group="Service", color=service_color, size=30, mass=5)
                    added_nodes.add(node_name)

            # --- MODIFIED EDGE HANDLING ---
            # Create a rich tooltip for the single aggregated edge
            rel_types = record.get("rel_types", [])
            rel_props_list = record.get("rel_props_list", [])

            # Create a unique list of relationship types for the edge label
            unique_rel_types = sorted(list(set(rel_types)))
            edge_label = ",\\n".join(unique_rel_types) # Display types on separate lines in the graph

            # Build a detailed title for the hover tooltip
            edge_title = "Interactions:\\n" + "="*20
            for i, rel_type in enumerate(rel_types):
                props = rel_props_list[i]
                details = props.get('details_text', 'No details')
                source_file = props.get('source_file', 'N/A')
                edge_title += f"\\n- Type: {rel_type}\\n  Details: {details[:80]}...\\n  Source: {source_file}\\n"

            # Add the single, aggregated edge to the graph
            if source_name != target_name:
                net.add_edge(source_name, target_name, title=edge_title, label=edge_label)

        file_name = "kg_aggregated_service_dependencies.html"
        net.save_graph(file_name)
        print(f"\nGraph visualization saved to {file_name}")

        display(HTML(net.generate_html()))
        print(f"If the graph is not displayed above, try opening '{file_name}' in your browser.")

    except Exception as e:
        print(f"An error occurred during visualization: {e}")
        traceback.print_exc()

# --- Execution part of the cell ---
if 'neo4j_graph' in globals() and neo4j_graph:
    if 'INFERRED_TOP_LEVEL_SERVICES' in globals() and INFERRED_TOP_LEVEL_SERVICES:
        visualize_aggregated_service_dependencies(neo4j_graph, INFERRED_TOP_LEVEL_SERVICES)
    else:
        print("Cannot visualize: INFERRED_TOP_LEVEL_SERVICES is not populated.")
else:
    print("Neo4j graph not initialized. Please run Cell 6 to initialize the connection.")

# @title New Cell: List Interactions Between Top-Level Services with Metadata Details
# Description: Queries Neo4j to list all relationships *between* services
#              that are in the INFERRED_TOP_LEVEL_SERVICES list, showing
#              the relationship type and all its properties (protocol, port, path, etc.).

from typing import List, Dict, Any, Optional # Ensure imports
import traceback # For error details
import json # For pretty printing properties if they are complex

def list_top_level_service_interactions(
    graph_db: Optional[Neo4jGraph],
    top_level_service_names: List[str], # This will be INFERRED_TOP_LEVEL_SERVICES
    limit: int = 250 # Adjustable limit for the number of relationships to display
):
    """
    Queries and prints relationships exclusively between the provided list of top-level service names,
    including all metadata (properties) stored on those relationships.
    """
    if not graph_db:
        print("Neo4j graph connection not available. Cannot list service interactions.")
        return
    if not top_level_service_names:
        print("The list of INFERRED_TOP_LEVEL_SERVICES is empty. Nothing to query.")
        return

    print(f"\n--- Listing Interactions Between {len(top_level_service_names)} Inferred Top-Level Services (Limited to {limit} Results) ---")
    if not isinstance(top_level_service_names, list) or not all(isinstance(item, str) for item in top_level_service_names):
        print(f"Error: top_level_service_names is not a list of strings. Value: {top_level_service_names}")
        return

    # Cypher query to find relationships *only between* the specified service nodes.
    # This shows directed relationships.
    query = """
    MATCH (s1:Service)-[r]->(s2:Service)
    WHERE s1.name IN $service_names AND s2.name IN $service_names
    RETURN
        s1.name AS source_service_name,
        type(r) AS relationship_type,
        properties(r) AS relationship_properties,
        s2.name AS target_service_name
    LIMIT $limit_val
    """
    query_params = {
        "service_names": top_level_service_names,
        "limit_val": limit
    }

    try:
        results = graph_db.query(query, params=query_params)

        if not results:
            print("No direct relationships found *between* the inferred top-level services in the graph.")
            return

        print(f"Found {len(results)} direct relationships between top-level services (displaying up to {limit}):")
        for i, record in enumerate(results):
            source_name = record.get('source_service_name', 'UnknownSourceService')
            rel_type = record.get('relationship_type', 'UNKNOWN_REL')
            rel_props = record.get('relationship_properties', {})
            target_name = record.get('target_service_name', 'UnknownTargetService')

            # Format properties for printing
            props_str_parts = []
            for k, v in rel_props.items():
                # Truncate very long string values for display brevity
                display_v = str(v)
                if len(display_v) > 70:
                    display_v = display_v[:67] + "..."
                props_str_parts.append(f"{k}: {repr(display_v)}") # Use repr to see quotes around strings

            props_display = ", ".join(props_str_parts)
            if props_display:
                props_display = f" {{{props_display}}}"
            else:
                props_display = "" # No properties on the relationship

            print(f"  {i+1}. (Service:{source_name}) "
                  f"-[:{rel_type}{props_display}]-> "
                  f"(Service:{target_name})")

    except Exception as e:
        print(f"Error querying for top-level service interactions: {e}")
        traceback.print_exc()

# --- Execution part of the cell ---
# This cell assumes 'neo4j_graph' (Cell 6) and
# 'INFERRED_TOP_LEVEL_SERVICES' (populated by Cell 3.6) are initialized.

if 'neo4j_graph' in globals() and neo4j_graph:
    if 'INFERRED_TOP_LEVEL_SERVICES' in globals() and INFERRED_TOP_LEVEL_SERVICES and isinstance(INFERRED_TOP_LEVEL_SERVICES, list):
        list_top_level_service_interactions(neo4j_graph, INFERRED_TOP_LEVEL_SERVICES, limit=500) # Increased limit
    else:
        # Attempt to re-populate INFERRED_TOP_LEVEL_SERVICES if not available or not a list
        print("INFO: INFERRED_TOP_LEVEL_SERVICES not populated correctly. Attempting to run service inference...")
        # This re-inference logic should ideally only run if Cell 3.6 hasn't.
        # It assumes the necessary functions (llm_identify_services_from_repo or infer_top_level_services_from_metadata)
        # and variables (MICROSERVICES_REPO_PATH, llm, etc.) are available.
        services_re_inferred = False
        if 'llm_identify_services_from_repo' in globals() and \
           'MICROSERVICES_REPO_PATH' in globals() and os.path.exists(MICROSERVICES_REPO_PATH) and \
           'llm' in globals() and llm and \
           'service_id_prompt_template' in globals() and service_id_prompt_template and \
           'service_id_parser' in globals() and service_id_parser:

            print("  Attempting LLM-based service re-inference...")
            # These assignments update the global variables
            IDENTIFIED_SERVICES_DETAILS = llm_identify_services_from_repo(MICROSERVICES_REPO_PATH, llm, service_id_prompt_template, service_id_parser)
            if IDENTIFIED_SERVICES_DETAILS:
                INFERRED_TOP_LEVEL_SERVICES = sorted(list(set([s.name.lower().replace("-", "_").replace(" ", "_").strip("._/") for s in IDENTIFIED_SERVICES_DETAILS if s.name and len(s.name)>2])))
                # Also re-populate INFERRED_SERVICE_DIRECTORY_PATTERNS if your normalization relies on it being fresh
                temp_patterns = {}
                for s_info in IDENTIFIED_SERVICES_DETAILS:
                    norm_s_name = s_info.name.lower().replace("-", "_").replace(" ", "_").strip("._/")
                    if not norm_s_name or norm_s_name not in INFERRED_TOP_LEVEL_SERVICES: continue
                    base_patterns = [f"src/{norm_s_name}/", f"cmd/{norm_s_name}/", f"apps/{norm_s_name}/", f"services/{norm_s_name}/", f"{norm_s_name}/"]
                    if s_info.source_directory_guess:
                        cleaned_source_dir_guess = s_info.source_directory_guess.strip('./\\').replace('\\', '/')
                        if cleaned_source_dir_guess and cleaned_source_dir_guess != norm_s_name: temp_patterns[norm_s_name] = [f"{cleaned_source_dir_guess}/"] + base_patterns
                        else: temp_patterns[norm_s_name] = base_patterns
                    elif norm_s_name : temp_patterns[norm_s_name] = base_patterns
                INFERRED_SERVICE_DIRECTORY_PATTERNS = temp_patterns
                services_re_inferred = True
                print(f"  Re-inferred services: {len(INFERRED_TOP_LEVEL_SERVICES)}")

        # Check again after attempt
        if 'INFERRED_TOP_LEVEL_SERVICES' in globals() and INFERRED_TOP_LEVEL_SERVICES and isinstance(INFERRED_TOP_LEVEL_SERVICES, list):
             list_top_level_service_interactions(neo4j_graph, INFERRED_TOP_LEVEL_SERVICES, limit=500)
        else:
            print("Cannot visualize: INFERRED_TOP_LEVEL_SERVICES is still empty or not a list after attempting inference.")
else:
    print("Neo4j graph not initialized. Please run Cell 6 to initialize the connection.")

# @title 14. Visualize and List CONNECTS_TO Relationships (Direction Agnostic)
# Description: Queries the CONNECTS_TO relationships between services and
#              generates a direction-agnostic (no arrows) visualization.

# Ensure pyvis is available
!pip -q install pyvis
from pyvis.network import Network
from IPython.core.display import display, HTML
PYVIS_AVAILABLE = True


from typing import Optional
import traceback

def visualize_and_list_connects_to_relationships(
    graph_db: Optional[Neo4jGraph],
    limit: int = 1000
):
    """
    Lists and then visualizes only the CONNECTS_TO relationships between
    service nodes, with direction-agnostic edges (no arrows).
    """
    if not graph_db:
        print("Neo4j graph connection not available.")
        return
    if not PYVIS_AVAILABLE:
        print("pyvis library not available. Cannot visualize.")
        return

    print(f"\n--- Analyzing Service-to-Service CONNECTS_TO Relationships ---")

    query = """
    MATCH (s1:Service)-[r:CONNECTS_TO]->(s2:Service)
    RETURN s1.name AS source, s2.name AS target, properties(r) AS props
    LIMIT $limit
    """
    query_params = {"limit": limit}

    try:
        results = graph_db.query(query, params=query_params)

        if not results:
            print(f"No CONNECTS_TO relationships between Service nodes were found.")
            return

        print(f"\n--- Textual List of CONNECTS_TO Relationships ---")
        for i, record in enumerate(results):
            source_node = record.get("source", "Unknown")
            target_node = record.get("target", "Unknown")
            rel_props = record.get("props", {})
            props_str = ", ".join([f"{k}: '{v}'" for k, v in rel_props.items() if v is not None and k != 'evidence'])
            if props_str:
                props_str = f" {{{props_str}}}"
            print(f"{i+1}. (Service:{source_node}) -[:CONNECTS_TO{props_str}]-> (Service:{target_node})")
        print("--------------------------------------------------\n")

        # MODIFIED: Set directed=False to make the graph direction-agnostic
        net = Network(notebook=True, cdn_resources='remote', height="800px", width="100%", directed=False)

        # MODIFIED: Removed the "arrows" configuration from the edges
        net.set_options("""
        var options = {
          "nodes": { "font": { "size": 16, "face": "Tahoma" }, "shape": "box" },
          "edges": { "font": { "size": 11, "align": "middle" }, "smooth": { "type": "dynamic" }},
          "physics": { "barnesHut": { "gravitationalConstant": -20000, "centralGravity": 0.1, "springLength": 250 }, "minVelocity": 0.75, "solver": "barnesHut"},
          "interaction": { "hover": true, "navigationButtons": true }
        }
        """)

        added_nodes = set()
        for record in results:
            source_node = record["source"]
            target_node = record["target"]
            rel_props = record.get("props", {})

            for node_name in [source_node, target_node]:
                if node_name not in added_nodes:
                    net.add_node(node_name, label=node_name, color="#87CEEB", size=30)
                    added_nodes.add(node_name)

            edge_title = "Type: CONNECTS_TO"
            props_str_tooltip = "\\n".join([f"{k}: {str(v)}" for k, v in rel_props.items()])
            if props_str_tooltip:
                edge_title += "\\n--- Properties ---\\n" + props_str_tooltip

            net.add_edge(
                source_node,
                target_node,
                title=edge_title,
                label="CONNECTS_TO",
                color="#FF6347",
                width=2.5
            )

        file_name = f"kg_connects_to_relationships_undirected.html"
        net.save_graph(file_name)
        print(f"\nUndirected graph visualization saved to {file_name}")

        display(HTML(net.generate_html()))
        print(f"If the graph is not displayed above, try opening '{file_name}' in your browser.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        traceback.print_exc()

# --- Execution ---
if 'neo4j_graph' in globals() and neo4j_graph:
    visualize_and_list_connects_to_relationships(neo4j_graph)
else:
    print("Neo4j graph connection not found. Please ensure the relevant cell has been run successfully.")

!pip -q install torch torch_geometric neo4j tqdm scikit-learn sentence-transformers

import os
import pandas as pd
from neo4j import GraphDatabase

# Set these in your environment or edit defaults:
# %env NEO4J_URI="neo4j://<host>:7687"
# %env NEO4J_USER="neo4j"
# %env NEO4J_PASSWORD="******"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

print(f"🔗 Connecting to {NEO4J_URI} as {NEO4J_USER}")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(cypher: str, params: dict | None = None) -> list[dict]:
    with driver.session() as session:
        res = session.run(cypher, params or {})
        return [r.data() for r in res]

# Node labels present in your KG (adjust as needed):

ALLOWED_PRIMARY_RELATIONSHIP_TYPES = {
    # Synchronous Communication
    "CONNECTS_TO", "EXPOSES_ENDPOINT", "EXPOSES_PORT",
    # Asynchronous Communication
    "PUBLISHES_TO", "SUBSCRIBES_TO",
    # Data Interaction
    "WRITES_TO", "READS_FROM",
    # General & Build-time
    "HAS_COMPONENT", "DEPENDS_ON", "WRITTEN_IN", "DEFINED_BY"
}

NODE_TYPES = [
    "Service", "Component", "APIGateway", "APIEndpoint", "Port",
    "MessageBroker", "MessageTopic", "Database",
    "ConfigurationFile", "TechnologyStack", "ProgrammingLanguage", "Tool"
]

# Relations that matter for documentation/retrieval (adjust as needed):
EDGE_TYPES_SCHEMA = [
    # --- Synchronous Communication ---
    # Services are the primary actors in communication
    ("Service", "CONNECTS_TO", "Service"),
    ("Service", "CONNECTS_TO", "APIGateway"),
    ("APIGateway", "CONNECTS_TO", "Service"),
    ("Component", "CONNECTS_TO", "Service"), # e.g., a gRPC client component

    # Gateways and Services expose endpoints and ports
    ("APIGateway", "EXPOSES_ENDPOINT", "APIEndpoint"),
    ("Service", "EXPOSES_ENDPOINT", "APIEndpoint"),
    ("Service", "EXPOSES_PORT", "Port"),

    # --- Asynchronous Communication ---
    # Services and Components can publish or subscribe
    ("Service", "PUBLISHES_TO", "MessageTopic"),
    ("Component", "PUBLISHES_TO", "MessageTopic"),
    ("Service", "SUBSCRIBES_TO", "MessageTopic"),
    ("Component", "SUBSCRIBES_TO", "MessageTopic"),

    # Topics are managed by a Broker
    ("MessageTopic", "DEPENDS_ON", "MessageBroker"),

    # --- Data Interaction ---
    # Services and Components interact with Databases
    ("Service", "WRITES_TO", "Database"),
    ("Service", "READS_FROM", "Database"),
    ("Component", "WRITES_TO", "Database"),
    ("Component", "READS_FROM", "Database"),

    # --- General & Build-time Relationships ---
    # Composition and high-level dependencies
    ("Service", "HAS_COMPONENT", "Component"),
    ("Service", "DEPENDS_ON", "Service"),
    ("Service", "DEPENDS_ON", "Database"),

    # Code and configuration level dependencies
    ("Component", "DEPENDS_ON", "TechnologyStack"),
    ("Service", "DEPENDS_ON", "TechnologyStack"),
    ("Service", "WRITTEN_IN", "ProgrammingLanguage"),
    ("Component", "WRITTEN_IN", "ProgrammingLanguage"),
    ("Service", "DEFINED_BY", "ConfigurationFile"),
    ("Component", "DEFINED_BY", "ConfigurationFile"),
    ("TechnologyStack", "DEFINED_BY", "ConfigurationFile"),

    # Tooling and build dependencies
    ("ConfigurationFile", "DEPENDS_ON", "Tool"), # e.g., a pom.xml depends on Maven
    ("Service", "DEPENDS_ON", "Tool"),
]

print("Planned node labels:", NODE_TYPES)
print("Planned relations:", EDGE_TYPES_SCHEMA)

# @title (FIXED) 4. Fetch Graph Data & Create Mappings
# Description: Fetches all nodes and edges from Neo4j.
# FIX: Uses the stable `elementId()` instead of the internal `id()`.
# This ensures mappings between graph data and embeddings are stable.

from collections import defaultdict
import pandas as pd

node_tables: dict[str, pd.DataFrame] = {}
# This map is stable: {ntype: {elementId_string: tensor_index_int}}
node_element_id_maps: dict[str, dict[str, int]] = {}

# Load nodes per label
print("Fetching node data from Neo4j...")
for ntype in NODE_TYPES:
    # Use elementId(n) for a stable, unique string ID
    # Also fetch props(n) to get 'name' and 'path' for the next step
    rows = run_query(f"MATCH (n:{ntype}) RETURN elementId(n) AS el_id, properties(n) AS props")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["el_id", "props"])
    if df.empty:
        print(f"⚠️ No nodes for label {ntype}")
        continue

    # Ensure props column exists even if some nodes have null properties
    if "props" not in df.columns:
        df["props"] = [{} for _ in range(len(df))]
    # Fill None props with empty dicts for safe access
    df["props"] = df["props"].apply(lambda x: x if isinstance(x, dict) else {})

    node_tables[ntype] = df
    # Create the map: stable `el_id` (str) -> new tensor row index `i` (int)
    node_element_id_maps[ntype] = {r.el_id: i for i, r in df.iterrows()}
    print(f"✅ {ntype}: {len(df)} nodes")

# Load edges per (src_label, rel, dst_label)
edge_tables: dict[tuple[str,str,str], pd.DataFrame] = {}
print("\nFetching edge data from Neo4j...")
for (s_t, r_t, d_t) in EDGE_TYPES_SCHEMA:
    # Fetch the stable elementId for source and destination
    rows = run_query(f"""
        MATCH (a:{s_t})-[r:{r_t}]->(b:{d_t})
        RETURN elementId(a) AS src_el_id, elementId(b) AS dst_el_id
    """)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["src_el_id", "dst_el_id"])
    edge_tables[(s_t, r_t, d_t)] = df
    print(f"✅ {s_t}-{r_t}->{d_t}: {len(df)} edges")

print("\n✅ Data fetch complete. node_element_id_maps created.")

# @title (FIXED) 5. Create HeteroData Object
# Description: Builds the PyG HeteroData object.
# FIX: Attaches the stable `element_id`, `name`, and `path` to each node type.
# FIX: Builds the edge_index tensor using the stable elementId maps.

import torch
!pip install torch_geometric
from torch_geometric.data import HeteroData

hetero_data = HeteroData()

# Add node data and metadata
print("Building HeteroData object...")
for ntype, df in node_tables.items():
    n = len(df)
    hetero_data[ntype].x = torch.zeros((n, 1), dtype=torch.float) # Placeholder features
    props_list = df["props"].tolist()

    # Store the stable IDs and key metadata on the graph object
    # This is the key to linking embeddings later
    hetero_data[ntype].element_id = df["el_id"].tolist()
    hetero_data[ntype].name = [p.get("name") for p in props_list]
    hetero_data[ntype].path = [p.get("path") for p in props_list]

# Add edges with local indices mapped from stable element IDs
for (s_t, r_t, d_t), edf in edge_tables.items():
    # Check if we have maps for both node types and if there are edges
    if s_t not in node_element_id_maps or d_t not in node_element_id_maps or edf.empty:
        continue

    # Use the stable map created in the previous cell
    s_map, d_map = node_element_id_maps[s_t], node_element_id_maps[d_t]
    s_idx, d_idx = [], []

    # Iterate over the edge dataframe
    for _, row in edf.iterrows():
        sid, did = row.src_el_id, row.dst_el_id
        # Add edge only if both source and destination element IDs were in our map
        if sid in s_map and did in d_map:
            s_idx.append(s_map[sid]) # Append the tensor index i
            d_idx.append(d_map[did]) # Append the tensor index j

    if len(s_idx) == 0:
        continue

    hetero_data[(s_t, r_t, d_t)].edge_index = torch.tensor([s_idx, d_idx], dtype=torch.long)

print("\nNode types:", hetero_data.node_types)
print("Edge types:", hetero_data.edge_types)
print("\n✅ PyG HeteroData object created with stable elementId mapping.")

from sentence_transformers import SentenceTransformer
import numpy as np
import torch # Ensure torch is imported
import re # For cleaning strings

# --- 1. Initialize Model ---
st_model = SentenceTransformer('all-MiniLM-L6-v2')
feature_dim = st_model.get_sentence_embedding_dimension()

# --- UPDATE 1: Updated props_to_text function ---
# This new function accepts the node's type (label) and includes 'evidence'.
def props_to_text(p: dict | None, ntype: str = "Unknown"):
    """
    Creates a single descriptive string from a node's properties,
    now including its type (label) and evidence.
    """
    if not isinstance(p, dict):
        p = {} # Use an empty dict for Nones

    # --- UPDATE 1a: The descriptive string now starts with the node's type (e.g., "type:service").
    buf = [f"type:{ntype.lower()}"]

    # --- UPDATE 1b: Added 'evidence' to the list of preferred properties to encode.
    # This list now includes 'evidence' to capture the LLM's reasoning.
    keys_pref = ["name", "path", "evidence", "description", "language", "title"]

    for k in keys_pref:
        v = p.get(k)
        if v:
            # Clean up the value string (remove newlines, strip whitespace)
            v_str = str(v).replace('\\n', ' ').replace('\n', ' ').strip()
            if v_str:
                buf.append(f"{k}:{v_str}")

    # Fallback: if we *only* have the type (i.e., no name, path, or evidence),
    # use the type itself as the descriptive text.
    if len(buf) == 1:
         buf.append(ntype.lower())

    return " ".join(buf)

print(f"Sentence Transformer loaded. Feature dimension is {feature_dim}.")
print(f"Enhanced props_to_text function is ready.")

# --- 2. Build Features for Node Types with Data ---
print("Generating features for populated node types...")
for ntype, df in node_tables.items():
    if df.empty:
        print(f"  - Skipping '{ntype}' as it has 0 nodes.")
        continue # This is correct, we will handle it in the next step

    print(f"  - Processing {len(df)} nodes for type '{ntype}'...")
    props_list = df["props"].tolist()

    # --- UPDATE 2: Pass 'ntype' (e.g., "Service") to the helper function. ---
    texts = [props_to_text(p, ntype) for p in props_list]

    # --- UPDATE 3: Removed the old fallback logic. ---
    # The new props_to_text function is robust and guarantees a
    # non-empty string, so the old fallback is no longer needed.
    # OLD: texts = [t if t.strip() else (hetero_data[ntype].name[i] or ntype) for i,t in enumerate(texts)]

    # Batch-encode to avoid OOM
    embs = st_model.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
    hetero_data[ntype].x = torch.tensor(embs, dtype=torch.float)

# --- 3. Assign Placeholders for Empty Node Types ---
# (This section is unchanged, but still critical)
print("\nAssigning placeholder features for empty node types...")
for ntype in hetero_data.node_types:
    if not hasattr(hetero_data[ntype], 'x'):
        print(f"  - Assigning placeholder for '{ntype}'.")
        # Create an empty tensor with the correct feature dimension
        hetero_data[ntype].x = torch.empty((0, feature_dim), dtype=torch.float)

# --- UPDATE 4: Changed print statement to reflect the update. ---
print("\n✅ Node feature generation complete with enhanced descriptions.")

# --- 4. Final Validation ---
print("\nNode feature dimensions:")
print({nt: tuple(hetero_data[nt].x.shape) for nt in hetero_data.node_types})

# --- 5. Example of a new text string ---
if 'Service' in node_tables and not node_tables['Service'].empty:
    sample_props = node_tables['Service']["props"].iloc[0]
    sample_text = props_to_text(sample_props, 'Service')
    print(f"\nExample initialization string for a Service node:\n'{sample_text}'")

# @title (FIXED) 7. Create Train/Val/Test Splits
# Description: Splits the graph data for link prediction.
# FIX: Lowered MIN_EDGES_FOR_SPLIT to ensure important relations
# like CONNECTS_TO are included in the training.

from torch_geometric.transforms import RandomLinkSplit

# Count edges per type
rel_counts = {et: hetero_data[et].edge_index.shape[1] for et in hetero_data.edge_types}
print("Edges per type:", rel_counts)

# FIX: Lowered from 20 to 10 to include 'CONNECTS_TO' (which had 16 edges)
MIN_EDGES_FOR_SPLIT = 10

EDGE_TYPES_FOR_TRAIN = [et for et, c in rel_counts.items() if c >= MIN_EDGES_FOR_SPLIT]
if not EDGE_TYPES_FOR_TRAIN:
    EDGE_TYPES_FOR_TRAIN = list(hetero_data.edge_types)
    print("⚠️ Using all edge types for training due to low counts.")
else:
    skipped_types = [et for et, c in rel_counts.items() if c < MIN_EDGES_FOR_SPLIT]
    if skipped_types:
        print(f"⚠️ Skipping relations with < {MIN_EDGES_FOR_SPLIT} edges: {skipped_types}")


print("\nTraining relations:", EDGE_TYPES_FOR_TRAIN)

split = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=False,
    add_negative_train_samples=True,
    edge_types=EDGE_TYPES_FOR_TRAIN,
)
train_data, val_data, test_data = split(hetero_data)
print("✅ Splits ready.")

# @title GNN Model Definition (FIXED)
# FIX: The model __init__ now accepts an `x_shapes` dictionary to handle
# different input feature dimensions (e.g., 384 for 'Service', 1 for 'Tool').
# This ensures the `lin_in` projection layers are created with the correct
# `in_features` dimension, fixing the matrix multiplication error.

import torch
import torch.nn as nn
import torch.nn.functional as F # Ensure F is imported
from torch_geometric.nn import HeteroConv, SAGEConv

# --- GNN Encoder Definition ---
class RobustHeteroSAGE(nn.Module):
    """
    The 2-Layer Heterogeneous GraphSAGE model,
    updated to handle heterogeneous input feature dimensions.
    """
    def __init__(self, metadata, x_shapes: dict[str, int], h: int, out: int, p_drop=0.2):
        super().__init__()

        # --- FIX: Add input projection layers based on x_shapes ---
        self.lin_in = nn.ModuleDict()
        for ntype in metadata[0]: # Iterate over all node types in metadata
            # Get the actual input feature dim for this ntype
            in_dim = x_shapes.get(ntype)

            if in_dim is None:
                print(f"⚠️ Warning: No features for ntype '{ntype}'. Initializing projection from dim 1.")
                in_dim = 1 # Fallback, though this shouldn't happen if x_shapes is correct

            self.lin_in[ntype] = nn.Linear(in_dim, h)

        # --- GNN Layers ---
        try:
            # All inputs to conv1 will now be projected to h-dim
            self.conv1 = HeteroConv({et: SAGEConv((h, h), h) for et in metadata[1]}, aggr='sum')
            # All inputs to conv2 will also be h-dim
            self.conv2 = HeteroConv({et: SAGEConv((h, h), out) for et in metadata[1]}, aggr='sum')
        except Exception as e:
            print(f"Error during HeteroConv initialization. Check metadata format.")
            print(f"Metadata node types: {metadata[0]}")
            print(f"Metadata edge types: {metadata[1]}")
            raise e

        self.drop = nn.Dropout(p_drop)
        print(f"Initialized 2-Layer HeteroSAGE with Input Projection (dims={x_shapes}), h={h}, out={out}.")

    def forward(self, x_dict, edge_index_dict):
        if not x_dict:
             print("Warning: Input x_dict to HeteroSAGE forward is empty.")
             return {}

        # --- Step 1: Apply Input Projection ---
        # Project all input features (e.g., dim 384 or 1) to dim h (256)
        x_proj = {}
        for ntype, x in x_dict.items():
            if x is not None and isinstance(x, torch.Tensor) and x.numel() > 0:
                try:
                    x_proj[ntype] = self.lin_in[ntype](x)
                except Exception as e:
                    print(f"Error projecting input for ntype {ntype} with shape {x.shape}")
                    raise e

        # x_proj is now a "dense" dict where all tensors have dim h (256)

        # --- Layer 1 ---
        # Pass the h-dim tensors to conv1
        x_out_1 = self.conv1(x_proj, edge_index_dict)

        # --- Create Input for Layer 2 ---
        x_in_2 = {}
        for ntype in x_proj.keys():
            h_conv1 = x_out_1.get(ntype)
            h_proj = x_proj[ntype]

            if h_conv1 is not None:
                # Node type was a destination, use its updated (and activated) embedding
                x_in_2[ntype] = self.drop(h_conv1.relu())
            else:
                # Node type was source-only, pass its original projected embedding
                x_in_2[ntype] = h_proj

        # --- Layer 2 ---
        # All tensors in x_in_2 now have dim h (256)
        x_out_2 = self.conv2(x_in_2, edge_index_dict)

        # --- Create Final Output ---
        final_x = {}
        for ntype in x_proj.keys():
            h_conv2 = x_out_2.get(ntype)
            if h_conv2 is not None:
                # This node type was a destination in conv2
                final_x[ntype] = h_conv2
            elif ntype in x_in_2:
                # Pass through the features from the previous layer
                # This ensures *all* nodes get an embedding
                final_x[ntype] = x_in_2[ntype]

        if not final_x:
            print("Warning: Final output dict from HeteroSAGE is empty.")

        return final_x

# --- Link Predictor Definition (No change needed) ---
class LinkPredictor(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, 1)
        )

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        src_embs = z_src[row]
        dst_embs = z_dst[col]
        h = torch.cat([src_embs, dst_embs], dim=-1)
        score = self.mlp(h).squeeze(-1)
        return score

print("✅ (FIXED) 2-Layer HeteroSAGE and LinkPredictor definitions ready.")

# @title Training/Evaluation Functions (Adjusted for Robust Model)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np # Make sure numpy is imported

# Helper packers (ensure they handle device correctly)
def _pack_x(data, device):
    # Only pack tensors that exist and are not empty
    return {k: v.to(device) for k,v in data.x_dict.items() if v is not None and v.numel() > 0}

def _pack_ei(data, device):
    # Only pack edge indices that exist
    return {k: v.to(device) for k,v in data.edge_index_dict.items() if v is not None}

def train_one_epoch(model, predictor, data, edge_types_to_train, optim, device):
    """ Trains the model for one epoch. """
    model.train()
    predictor.train()
    optim.zero_grad()

    valid_x = _pack_x(data, device)
    valid_ei = _pack_ei(data, device)

    if not valid_x:
         print("Warning: No valid node features found in train_one_epoch input.")
         return 0.0

    z = model(valid_x, valid_ei) # Call the RobustHeteroSAGE model

    preds, gts = [], []
    valid_edge_types_processed = 0
    for et in edge_types_to_train:
        if et not in data.edge_index_dict or not hasattr(data[et], 'edge_label_index') or not hasattr(data[et], 'edge_label'):
             continue

        src, _, dst = et
        # Check if embeddings exist for BOTH src and dst in the model output z
        if src in z and dst in z:
            edge_label_index = data[et].edge_label_index.to(device)
            edge_label = data[et].edge_label.to(device)
            if edge_label_index.numel() > 0:
                 preds.append(predictor(z[src], z[dst], edge_label_index))
                 gts.append(edge_label)
                 valid_edge_types_processed += 1

    if not preds:
         print("Warning: No valid predictions made in train_one_epoch. Check model output z and edge_types_to_train.")
         return 0.0

    try:
        pred = torch.cat(preds, dim=0)
        gt = torch.cat(gts, dim=0)
        loss = F.binary_cross_entropy_with_logits(pred, gt)
        loss.backward()
        optim.step()
        return loss.item()
    except Exception as e:
        print(f"Error during loss calculation or backward pass: {e}")
        return float('nan')


@torch.no_grad()
def evaluate(model, predictor, data, edge_types_to_eval, device):
    """ Evaluates the model on validation or test data. """
    model.eval()
    predictor.eval()

    valid_x = _pack_x(data, device)
    valid_ei = _pack_ei(data, device)

    if not valid_x:
         print("Warning: No valid node features found in evaluate input.")
         return 0.0, 0.0, float('inf')

    z = model(valid_x, valid_ei) # Call the RobustHeteroSAGE model

    preds, gts = [], []
    total_loss = 0.0
    num_eval_types = 0
    valid_edge_types_processed = 0

    for et in edge_types_to_eval:
        if et not in data.edge_index_dict or not hasattr(data[et], 'edge_label_index') or not hasattr(data[et], 'edge_label'):
            continue

        src, _, dst = et
        if src in z and dst in z: # Check if embeddings exist for BOTH src and dst
            edge_label_index = data[et].edge_label_index.to(device)
            edge_label = data[et].edge_label.to(device)
            if edge_label_index.numel() > 0 and edge_label.numel() > 0:
                pred = predictor(z[src], z[dst], edge_label_index)
                gt = edge_label
                preds.append(pred)
                gts.append(gt)
                total_loss += F.binary_cross_entropy_with_logits(pred, gt).item()
                num_eval_types += 1
                valid_edge_types_processed += 1

    if not preds:
         print("Warning: No valid evaluations made in evaluate function. Check model output z and edge_types_to_eval.")
         return 0.0, 0.0, float('inf')

    avg_loss = total_loss / num_eval_types if num_eval_types > 0 else float('inf')

    try:
        pred = torch.cat(preds, dim=0)
        gt = torch.cat(gts, dim=0)
        pred_sigmoid = torch.sigmoid(pred).cpu().numpy()
        gt_numpy = gt.cpu().numpy()
    except Exception as e:
        print(f"Error during final eval tensor concatenation or conversion: {e}")
        return 0.0, 0.0, avg_loss

    if len(np.unique(gt_numpy)) < 2:
        auc = 0.5
        ap = np.mean(gt_numpy) if len(gt_numpy) > 0 else 0.5
    else:
        try:
            auc = roc_auc_score(gt_numpy, pred_sigmoid)
            ap = average_precision_score(gt_numpy, pred_sigmoid)
        except ValueError as e:
            print(f"Warning: Error calculating AUC/AP: {e}. Returning default values.")
            auc = 0.5
            ap = 0.5

    return auc, ap, avg_loss

print("✅ Training and evaluation functions ready (adjusted for Robust Model).")

# @title GNN Model Training (FIXED)
# FIX 1: Detects the *actual* input feature dimensions (e.g., 384, 1)
#        from `hetero_data` and saves them in `x_shapes`.
# FIX 2: Passes `x_shapes` to the RobustHeteroSAGE constructor.
# FIX 3: Saves `x_shapes` to the checkpoint file for correct model loading.

import time, torch, os

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Hyperparameters ---
hidden_dim = 256
out_dim = 256
lr = 1e-3
weight_decay = 1e-5
epochs = 50
patience = 10

# --- Move Data Objects to Device ---
if 'train_data' in globals() and train_data:
    train_data = train_data.to(device)
    print("Moved train_data to device.")
if 'val_data' in globals() and val_data:
    val_data = val_data.to(device)
    print("Moved val_data to device.")
if 'test_data' in globals() and test_data:
    test_data = test_data.to(device)
    print("Moved test_data to device.")

# Check if data objects exist before proceeding
if 'train_data' not in globals() or not train_data:
     print("❌ train_data not found or is empty. Cannot proceed with training.")
else:
    # --- FIX 1: Get the actual feature shapes ---
    x_shapes = None
    if 'hetero_data' in globals() and hetero_data:
        # Get the actual input feature dimensions for each node type
        x_shapes = {ntype: x.shape[1] for ntype, x in hetero_data.x_dict.items() if x is not None and x.numel() > 0}
        print(f"✅ Detected input feature shapes: {x_shapes}")
    else:
        print("❌ Full 'hetero_data' object not found. Cannot determine feature shapes.")
    # --- End of FIX 1 ---

    # --- Model Initialization using RobustHeteroSAGE ---
    try:
        if 'hetero_data' in globals() and hetero_data and x_shapes:
             # --- FIX 2: Pass x_shapes to constructor ---
             model = RobustHeteroSAGE(hetero_data.metadata(), x_shapes, hidden_dim, out_dim, p_drop=0.2).to(device)
        else:
             print("❌ Full 'hetero_data' or 'x_shapes' not found. Cannot initialize RobustHeteroSAGE.")
             model = None

        predictor = LinkPredictor(out_dim).to(device)
        optim = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr, weight_decay=weight_decay)
        print("Model, Predictor, and Optimizer initialized.")

    except Exception as e:
        print(f"❌ Error initializing model/predictor: {e}")
        import traceback
        traceback.print_exc()
        model = None

    # --- Training Loop ---
    if model and predictor and optim:
        best_val = float('inf')
        best_path = "./gnn_hetero_best_model_robust.pth" # New save path
        stale = 0
        training_successful = False

        print("\n--- Starting Training Loop (with RobustHeteroSAGE model) ---")
        for ep in range(1, epochs+1):
            t0 = time.time()
            try:
                if 'train_data' not in globals() or not train_data or \
                   'val_data' not in globals() or not val_data:
                     print("❌ Data objects missing during training loop. Aborting.")
                     break

                tr = train_one_epoch(model, predictor, train_data, EDGE_TYPES_FOR_TRAIN, optim, device)
                if torch.isnan(torch.tensor(tr)): # Check if loss became NaN
                    print(f"❌ Training Loss is NaN at epoch {ep}. Stopping.")
                    break
                va_auc, va_ap, va = evaluate(model, predictor, val_data, EDGE_TYPES_FOR_TRAIN, device)
                dt = time.time() - t0
                print(f"Epoch {ep:03d} | {dt:.2f}s | train={tr:.4f} | val_loss={va:.4f} | val_auc={va_auc:.4f} | val_ap={va_ap:.4f}")

                if va < best_val - 1e-4:
                    best_val = va
                    stale = 0
                    # --- FIX 3: Save x_shapes to checkpoint ---
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "metadata": hetero_data.metadata(), # Save metadata
                        "x_shapes": x_shapes,                # <-- ADD THIS
                        "hidden_dim": hidden_dim, "out_dim": out_dim
                    }, best_path)
                else:
                    stale += 1
                    if stale >= patience:
                        print(f"⏹️ Early stopping at epoch {ep} (best val_loss={best_val:.4f})")
                        training_successful = True
                        break
                if ep == epochs:
                     training_successful = True

            except Exception as e:
                print(f"❌ ERROR during training epoch {ep}: {e}")
                import traceback
                traceback.print_exc()
                break

        # --- Final Evaluation ---
        if training_successful and os.path.exists(best_path):
            print("\n--- Loading best model (RobustHeteroSAGE) for Test Set Evaluation ---")
            try:
                ckpt = torch.load(best_path, map_location=device)
                # Re-initialize the model with the *saved* metadata and x_shapes
                model = RobustHeteroSAGE(ckpt["metadata"], ckpt["x_shapes"], ckpt["hidden_dim"], ckpt["out_dim"]).to(device)
                predictor = LinkPredictor(ckpt["out_dim"]).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                predictor.load_state_dict(ckpt["predictor_state_dict"])

                if 'test_data' in globals() and test_data:
                    te_auc, te_ap, te = evaluate(model, predictor, test_data, EDGE_TYPES_FOR_TRAIN, device)
                    print(f"\n✅ Test — loss={te:.4f} | AUC={te_auc:.4f} | AP={te_ap:.4f}")
                else:
                    print("⚠️ test_data not found. Cannot perform final test evaluation.")

            except Exception as e:
                print(f"❌ Error loading best model or evaluating on test set: {e}")
                import traceback
                traceback.print_exc()
        elif not os.path.exists(best_path):
             print("\n❌ Training completed/stopped, but best model file was not saved.")
        else:
             print("\n❌ Training did not complete successfully.")
    else:
        print("❌ Model/Predictor/Optimizer not initialized correctly. Skipping training loop.")

# @title (FIXED) 11. Generate and Save Final Node Embeddings
# Description: Encodes all nodes in the graph using the trained model.
# FIX: Saves the embeddings as a dictionary {element_id: vector}
# to create a stable lookup for the reranking pipeline.

@torch.no_grad()
def encode_full(hetero_d, model, device='cpu'):
    model.eval()
    # Pass the full graph's features and edge indices to the model
    return model(_pack_x(hetero_d, device), _pack_ei(hetero_d, device))

# Get the final embeddings for all nodes in the original graph
# Use the 'model' variable from the successful training cell
z_tensors = encode_full(hetero_data, model, device=device)

# --- This is the critical fix ---
# Create a dictionary mapping the STABLE element_id to the generated vector
z_dict_mapped = {}
for ntype, tensor in z_tensors.items():
    if ntype not in hetero_data.node_types: continue # Safety check
    tensor = tensor.detach().cpu()
    # Get the list of element_ids we stored on the graph object in cell 5
    element_ids = hetero_data[ntype].element_id

    # Create a new dict for this node type
    z_dict_mapped[ntype] = {}
    for i, el_id in enumerate(element_ids):
        z_dict_mapped[ntype][el_id] = tensor[i] # map '4:uuid:1' -> tensor[0]
# --- End of fix ---

torch.save(z_dict_mapped, "./z_dict.pt")
print(f"✅ Saved node embeddings as an element_id->vector map to ./z_dict.pt")

# Print a sample to verify structure
sample_ntype = 'Service'
if sample_ntype in z_dict_mapped and z_dict_mapped[sample_ntype]:
    sample_id = next(iter(z_dict_mapped[sample_ntype].keys()))
    print(f"\nSample embedding structure for '{sample_ntype}':")
    print(f"  '{sample_id}': {tuple(z_dict_mapped[sample_ntype][sample_id].shape)}")

torch.save(hetero_data, "./hetero_graph_data.pt")
print("✅ Saved hetero graph to ./hetero_graph_data.pt")



# KG quality Evaluation

# @title Enrich Graph: Add Inferred CONNECTS_TO Relationships
# Description: This cell finds services connected indirectly via a shared
#              message topic and enriches the graph by adding a direct
#              CONNECTS_TO relationship between them.

from typing import Optional, List
import traceback

def enrich_graph_with_inferred_connections(
    graph_db: Optional[Neo4jGraph],
    inferred_services: List[str]
):
    """
    Finds services that publish/subscribe to the same topic and creates a direct
    CONNECTS_TO relationship between them, adding provenance properties.
    """
    if not graph_db:
        print("Neo4j graph connection not available.")
        return
    if not inferred_services:
        print("Inferred services list is empty. Cannot perform enrichment.")
        return

    print(f"\n--- Enriching graph with inferred CONNECTS_TO relationships from pub/sub patterns ---")

    # This Cypher query finds the indirect connection pattern, then uses MERGE
    # to create a single, direct CONNECTS_TO relationship between the services.
    # `WITH DISTINCT` ensures we only create one relationship per service pair,
    # even if they communicate over multiple topics.
    enrichment_query = """
    MATCH (p_service:Service)-[:HAS_COMPONENT*0..1]->(publisher)-[:PUBLISHES_TO]->(topic:MessageTopic)
    WHERE p_service.name IN $service_list

    MATCH (s_service:Service)-[:HAS_COMPONENT*0..1]->(subscriber)-[:SUBSCRIBES_TO]->(topic:MessageTopic)
    WHERE s_service.name IN $service_list AND p_service <> s_service

    WITH DISTINCT p_service, s_service, collect(topic.name) as via_topics

    MERGE (p_service)-[r:CONNECTS_TO]->(s_service)
    ON CREATE SET
        r.reason = 'Inferred from pub/sub connection',
        r.via_topics = via_topics,
        r.inferred_at = datetime()
    RETURN
        p_service.name AS publisher,
        s_service.name AS subscriber,
        via_topics
    """

    try:
        results = graph_db.query(enrichment_query, params={"service_list": inferred_services})

        if results:
            print(f"✅ Successfully created or merged {len(results)} inferred CONNECTS_TO relationships.")
            for record in results:
                print(f"  - Inferred: ({record['publisher']}) -[:CONNECTS_TO]-> ({record['subscriber']}) via topics: {record['via_topics']}")
        else:
            print("No new indirect relationships were found to infer connections from.")

    except Exception as e:
        print(f"An error occurred during graph enrichment: {e}")
        traceback.print_exc()

# --- Execution ---
if 'neo4j_graph' in globals() and 'INFERRED_TOP_LEVEL_SERVICES' in globals():
    enrich_graph_with_inferred_connections(
        neo4j_graph,
        INFERRED_TOP_LEVEL_SERVICES
    )
else:
    print("❌ Prerequisites not met. Please ensure 'neo4j_graph' and 'INFERRED_TOP_LEVEL_SERVICES' are available.")

# @title Final KG Evaluation Pipeline (with Detailed Reporting)
# Description: This version provides detailed lists of True/False Positives
#              and Negatives for both global edges and per-node neighbors.

# --- 1. Imports and Setup ---
!pip install networkx textdistance -q

import networkx as nx
import textdistance
import re
from typing import Set, Dict, Any, Tuple, List
from langchain_community.graphs import Neo4jGraph

print("Libraries installed and imported.")


# --- 2. Ground Truth & Generated Graph Builders ---
# 'admin_server', 'api_gateway', 'config_server', 'customers', 'discovery_server', 'genai', 'grafana', 'prometheus', 'vets', 'visits'
      # (api_gateway) <--> (customer)
      # (api_gateway) <--> (genai)
      # (api_gateway) <--> (vets)
      # (api_gateway) <--> (visits)
      # (customer) <--> (genai)
def build_ground_truth_graph() -> nx.DiGraph:
    """Creates and returns the ground truth networkx graph."""
    ground_truth_graph = nx.DiGraph()
    ground_truth_graph.add_nodes_from([
        ("frontend", {"entity_type": "Service"}),
        ("userservice", {"entity_type": "Service"}),
        ("contacts", {"entity_type": "Service"}),
        ("ledger_writer", {"entity_type": "Service"}),
        ("balance_reader", {"entity_type": "Service"}),
        ("transaction_history", {"entity_type": "Service"}),
        ("load_generator", {"entity_type": "Service"}),
    ])
    edges_gt = [
        ("frontend", "userservice"),
        ("frontend", "contacts"),
        ("frontend", "ledger_writer"),
        ("frontend", "balance_reader"),
        ("frontend", "transaction_history"),
        ("ledger_writer", "balance_reader"),
    ]
    ground_truth_graph.add_edges_from(edges_gt)
    print("\n--- Ground Truth Graph ---")
    print(f"Nodes: {ground_truth_graph.number_of_nodes()}, Edges: {ground_truth_graph.number_of_edges()}")
    return ground_truth_graph

def build_generated_graph_from_db(graph_db: Neo4jGraph) -> nx.DiGraph:
    """
    Queries the live Neo4j database to build a networkx graph of the
    final service-level architecture after all enrichment steps.
    """
    print("\n--- Building Generated Graph from Final State in Neo4j Database ---")
    generated_graph = nx.DiGraph()

    nodes_query = "MATCH (n:Service) RETURN n.name AS name, 'Service' AS entity_type"
    node_results = graph_db.query(nodes_query)
    for record in node_results:
        generated_graph.add_node(record['name'], entity_type=record['entity_type'])

    edges_query = """
    MATCH (s1:Service)-[:CONNECTS_TO]->(s2:Service)
    RETURN s1.name AS source, s2.name AS target
    """
    edge_results = graph_db.query(edges_query)
    for record in edge_results:
        generated_graph.add_edge(record['source'], record['target'])

    print(f"Generated Graph built from DB.")
    print(f"Nodes: {generated_graph.number_of_nodes()}, Edges: {generated_graph.number_of_edges()}")
    return generated_graph


# --- 3. The Comprehensive Evaluation Program ---

def normalize_name_robust(name: str) -> str:
    """Creates a robust canonical name for comparison."""
    if not name: return ""
    return name.lower().replace('-', '_').replace('_management', '').replace('service', '').strip()

def find_node_anchors_hybrid(gen_graph: nx.DiGraph, gt_graph: nx.DiGraph, threshold: float) -> Tuple[Dict, Set, Set]:
    """Finds matching nodes using a hybrid of normalization and edit distance."""
    gt_nodes_data = {node: data for node, data in gt_graph.nodes(data=True)}
    gen_nodes_data = {node: data for node, data in gen_graph.nodes(data=True)}
    unmatched_gt, unmatched_gen, anchor_map = list(gt_nodes_data.keys()), list(gen_nodes_data.keys()), {}

    while unmatched_gen and unmatched_gt:
        best_score, best_pair = -1, None
        for g_node in unmatched_gen:
            for gt_node in unmatched_gt:
                norm_g_name = normalize_name_robust(g_node)
                norm_gt_name = normalize_name_robust(gt_node)
                score = textdistance.levenshtein.normalized_similarity(norm_g_name, norm_gt_name)
                if score > best_score:
                    best_score, best_pair = score, (g_node, gt_node)

        if best_score >= threshold:
            g_node, gt_node = best_pair
            anchor_map[g_node] = gt_node
            unmatched_gen.remove(g_node)
            unmatched_gt.remove(gt_node)
        else:
            break

    return anchor_map, set(unmatched_gen), set(unmatched_gt)

def evaluate_knowledge_graph_comprehensive(generated_graph: nx.DiGraph, ground_truth_graph: nx.DiGraph):
    """Calculates and prints all three quality metrics with detailed triplet reporting."""
    print("\n\n" + "="*60)
    print("      COMPREHENSIVE KNOWLEDGE GRAPH QUALITY EVALUATION")
    print("="*60)

    # --- METRIC 1: Node Identification ---
    print("\n--- METRIC 1: Node Identification Accuracy ---")
    anchor_map, fp_nodes, fn_nodes = find_node_anchors_hybrid(generated_graph, ground_truth_graph, 0.85)
    tp = len(anchor_map)
    fp = len(fp_nodes)
    fn = len(fn_nodes)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1_score:.2f}")
    print(f"  - True Positives (Correctly Identified): {tp}")
    print(f"  - False Positives (Generated but not in GT): {fp} {fp_nodes or ''}")
    print(f"  - False Negatives (In GT but not Generated): {fn} {fn_nodes or ''}")

    # --- METRIC 2: Edge Identification ---
    print("\n\n--- METRIC 2: Global Edge Accuracy (Direction-Agnostic F1-Score) ---")
    remapped_gen_edges = {tuple(sorted((anchor_map[u], anchor_map[v]))) for u, v in generated_graph.edges() if u in anchor_map and v in anchor_map}
    gt_edges = {tuple(sorted(edge)) for edge in ground_truth_graph.edges()}
    tp_edges_set = gt_edges.intersection(remapped_gen_edges)
    fp_edges_set = remapped_gen_edges - gt_edges
    fn_edges_set = gt_edges - remapped_gen_edges
    precision_edge = len(tp_edges_set) / (len(tp_edges_set) + len(fp_edges_set)) if (len(tp_edges_set) + len(fp_edges_set)) > 0 else 0
    recall_edge = len(tp_edges_set) / (len(tp_edges_set) + len(fn_edges_set)) if (len(tp_edges_set) + len(fn_edges_set)) > 0 else 0
    f1_score_edge = 2 * (precision_edge * recall_edge) / (precision_edge + recall_edge) if (precision_edge + recall_edge) > 0 else 0
    print(f"Precision: {precision_edge:.2f} | Recall: {recall_edge:.2f} | F1-Score: {f1_score_edge:.2f}")
    print(f"\n  - True Positives (Correctly Identified Edges): {len(tp_edges_set)}")
    for u, v in sorted(list(tp_edges_set)): print(f"      ({u}) <--> ({v})")
    print(f"\n  - False Positives (Generated but not in GT): {len(fp_edges_set)}")
    for u, v in sorted(list(fp_edges_set)): print(f"      ({u}) <--> ({v})")
    print(f"\n  - False Negatives (Missed in GT but not Generated): {len(fn_edges_set)}")
    for u, v in sorted(list(fn_edges_set)): print(f"      ({u}) <--> ({v})")

    # --- METRIC 3: Per-Node Neighbor Accuracy ---
    print("\n\n--- METRIC 3: Per-Node One-Hop Accuracy (Jaccard Similarity & Detailed Neighbors) ---")
    node_jaccard_scores = {}
    print("\nBreakdown of Neighbors for each Anchored Node:")
    for gen_node, gt_node in sorted(anchor_map.items()):
        gen_neighbors_raw = set(generated_graph.predecessors(gen_node)).union(set(generated_graph.successors(gen_node)))
        gen_neighbors_mapped = {anchor_map.get(n) for n in gen_neighbors_raw if n in anchor_map}
        gt_neighbors_raw = set(ground_truth_graph.predecessors(gt_node)).union(set(ground_truth_graph.successors(gt_node)))

        tp_neighbors = gen_neighbors_mapped.intersection(gt_neighbors_raw)
        fp_neighbors = gen_neighbors_mapped - gt_neighbors_raw
        fn_neighbors = gt_neighbors_raw - gen_neighbors_mapped

        intersection = len(tp_neighbors)
        union = len(tp_neighbors) + len(fp_neighbors) + len(fn_neighbors)
        score = intersection / union if union > 0 else 1.0
        node_jaccard_scores[gt_node] = score

        print(f"\n  - Node '{gt_node}' (Jaccard Score: {score:.2f}):")
        print(f"    - True Positives (Correct Neighbors): {len(tp_neighbors)} {sorted(list(tp_neighbors)) if tp_neighbors else ''}")
        print(f"    - False Positives (Incorrect Neighbors): {len(fp_neighbors)} {sorted(list(fp_neighbors)) if fp_neighbors else ''}")
        print(f"    - False Negatives (Missed Neighbors): {len(fn_neighbors)} {sorted(list(fn_neighbors)) if fn_neighbors else ''}")

    if node_jaccard_scores:
        average_jaccard = sum(node_jaccard_scores.values()) / len(node_jaccard_scores)
        print(f"\nAverage Neighbor Similarity (Jaccard Index): {average_jaccard:.2f}")
    else:
        print("No anchored nodes to evaluate for neighbor similarity.")
    print("="*60)


# --- 5. Main Execution Block ---
if 'neo4j_graph' in globals() and neo4j_graph:
    gt_graph = build_ground_truth_graph()
    gen_graph = build_generated_graph_from_db(neo4j_graph)
    evaluate_knowledge_graph_comprehensive(gen_graph, gt_graph)
else:
    print("❌ Neo4j graph connection not found. Please ensure the relevant cell has been run successfully.")

# @title 19. Final Pipeline: Generate Full Documentation from Knowledge Graph
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from typing import Optional

# --- Step 1: Function to Create the Graph Representation ---
# This function creates a detailed adjacency list from the graph.
def generate_adjacency_list_representation(graph_db: Optional[Neo4jGraph]) -> str:
    if not graph_db: return "Graph database connection not available."
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    WITH n, r, m
    ORDER BY n.name, type(r)
    RETURN
      n.name AS source_name,
      labels(n)[0] AS source_label,
      type(r) AS rel_type,
      properties(r) AS rel_props,
      m.name AS target_name,
      labels(m)[0] AS target_label
    """
    results = graph_db.query(query)
    adj_list = {}
    for record in results:
        source_name, source_label = record.get("source_name"), record.get("source_label", "Unknown")
        if not source_name: continue
        if source_name not in adj_list:
            adj_list[source_name] = {"label": source_label, "relationships": []}
        if record.get("rel_type"):
            props = record.get("rel_props", {})
            props_str = ""
            if props:
                # Include properties like 'reason' or 'via_topics' for enriched descriptions
                props_str = " {" + ", ".join([f"{k}:'{v}'" for k,v in props.items() if k != 'evidence']) + "}"
            rel_str = f"-[:{record['rel_type']}{props_str}]-> ({record.get('target_name', 'Unknown')}:{record.get('target_label', 'Unknown')})"
            adj_list[source_name]["relationships"].append(rel_str)

    representation_parts = []
    for name, data in sorted(adj_list.items()):
        representation_parts.append(f"NODE: {name} (Type: {data['label']})")
        if data['relationships']:
            representation_parts.append("  RELATIONSHIPS:")
            for rel_str in sorted(data['relationships']):
                representation_parts.append(f"    {rel_str}")
        representation_parts.append("---")
    return "\n".join(representation_parts)

# --- Step 2: Main Execution Block ---
print("--- Starting Final Documentation Generation Pipeline ---")
if 'neo4j_graph' in globals() and neo4j_graph and 'llm' in globals() and llm:

    # 1. Generate the detailed graph representation to use as context
    print("\n[1/3] Generating graph representation from Neo4j...")
    graph_context = generate_adjacency_list_representation(neo4j_graph)
    print("✅ Graph representation generated.")

    # 2. Define the final, "Graph-to-Text" prompt
    # The new, more restrictive prompt template
    documentation_prompt = f"""You are an expert technical writer, but for this task, your primary directive is to act as a **strict graph-to-text converter**. Your sole source of truth is the provided KNOWLEDGE GRAPH REPRESENTATION and the list of TOP-LEVEL SERVICES.

---
**CRITICAL RULES - YOU MUST FOLLOW THESE:**

1.  **NO INFERENCE OR INVENTION:** You MUST NOT invent, infer, or assume any service, component, function, or file name that is not **explicitly present** in the KNOWLEDGE GRAPH REPRESENTATION. If a detail is not in the provided context, it CANNOT be in your output.
2.  **USE EXACT NAMES:** Every service, component, tool, technology, or file name you mention MUST be one of the **exact names** found in the provided context.
3.  **USE BACKTICKS:** To ensure compliance, you MUST enclose every single entity name from the graph in backticks. For example: `checkoutservice`, `redis_cart`, `main_go`. This is a mandatory formatting requirement.
---

CONTEXTUAL INFORMATION:
The following is the complete list of top-level services that have been identified in the system:
{INFERRED_TOP_LEVEL_SERVICES}
---
KNOWLEDGE GRAPH REPRESENTATION (ADJACENCY LIST):
---
{graph_context}
---

TASK:
Write a full architecture document in Markdown format. You must adhere STRICTLY to the CRITICAL RULES above and base your writing **only** on the information provided in the context. Follow the specified document structure precisely.

**DOCUMENT STRUCTURE:**

# 1. Introduction
Write a brief, high-level summary of the system's architecture. Use the provided list of top-level services as the basis for your summary, ensuring every service mentioned is from the list and enclosed in backticks.

# 2. High-Level Architecture
Describe the main services and how they are connected at a high level. Use only the significant relationships between the top-level services (such as `CONNECTS_TO`, `DEPENDS_ON`, `PUBLISHES_TO`) to explain the primary communication and dependency paths. Remember to use exact names in backticks.

# 3. Service Details
Create a subsection for **each and every service** from the provided TOP-LEVEL SERVICES list. In each subsection:
- Briefly describe the service's purpose, as can be directly inferred from its name and relationships.
- List its main responsibilities by describing the exact components it has (using the `HAS_COMPONENT` relationship). **Do not invent function names.** If the component is `main_go`, state that its responsibilities are handled by the `main_go` component.
- Describe its key dependencies using the exact names from the `DEPENDS_ON` relationship.
- List its key components using the exact names from the `HAS_COMPONENT` relationship.
- Mention the programming language from the `WRITTEN_IN` relationship.

# 4. API & Communication View
Describe the key communication patterns in the system by analyzing the graph, adhering to the critical rules.

- **API Gateway Pattern:** Identify any `:APIGateway` nodes. Describe how the gateway routes requests by listing its `CONNECTS_TO` relationships to other services. List the primary `:APIEndpoint` nodes it `EXPOSES_ENDPOINT`.

- **Service Discovery:** Identify a central service discovery mechanism by looking for a `:Service` that many other services `DEPENDS_ON`.

- **Direct Synchronous Communication:** Detail any direct `CONNECTS_TO` relationships *between* services that do not involve an API Gateway.

- **Asynchronous Messaging:** Identify the central `:MessageBroker`. For each `:MessageTopic`, explain which services or components `PUBLISHES_TO` it and which `SUBSCRIBES_TO` it.

# 5. Data Architecture & Persistence View
Describe how services interact with databases. For each `:Service` that has a `WRITES_TO` or `READS_FROM` relationship to a `:Database` node, describe this specific interaction.

# 6. Deployment & Infrastructure View
Describe the infrastructure and configuration.
- **Service Definition:** Use the `DEFINED_BY` relationship to explain which `:ConfigurationFile` (e.g., `pom_xml`) defines which services or components.
- **Technology Stack:** For each `:TechnologyStack` found, explain its usage based on `DEPENDS_ON` relationships.
- **Exposed Ports:** For services that `EXPOSES_PORT`, state the port number.

# 7. Cross-Cutting Concerns
Based only on the available entities and relationships, briefly mention any observable cross-cutting concerns. For example, if you see a `jwt` or `jaeger` `:TechnologyStack` node, mention its connections.

# 8. Technology Stack and Tools
List the technologies and tools used. For each `:TechnologyStack` and `:Tool`, describe its relationship with other nodes. You must include all `:TechnologyStack` and `:Tool` nodes from the context.

Write the full document now based on these strict instructions.
"""

    # 3. Generate the final documentation
    print("\n[2/3] Sending request to LLM to generate the full documentation...")
    try:
        final_documentation = llm.invoke(documentation_prompt).content
        print("✅ Documentation generated successfully.")

        print("\n[3/3] Displaying Final Architecture Document:")
        print("="*60)
        # Display the final output
        print(final_documentation)
        print("="*60)

    except Exception as e:
        print(f"❌ An error occurred during documentation generation: {e}")

else:
    print("❌ Prerequisites not met. Please ensure 'neo4j_graph' and 'llm' objects are initialized.")

# NEW SECTION, GNN BASED RERANKING

# @title (FINAL FIXED) GNN Reranking Pipeline
# Description: A complete, self-contained reranking pipeline.
# FIX 1 (Critical): Loads the stable element_id->vector map from "./z_dict.pt".
# FIX 2 (Critical): Builds stable name_to_elementId and path_to_elementId maps.
# FIX 3 (Insufficiency): Implements robust n-gram entity linking for natural language queries.
# FIX 4 (Syntax): Removed stray backslash in _run_query method.

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import torch.nn.functional as F
import pandas as pd
from neo4j import GraphDatabase
import faiss
import json
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import textwrap
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("✅ All necessary libraries imported.")


# =============================================================================
# CONFIGURATION
# =============================================================================
class RerankerConfig:
    """Holds all configurable paths and parameters."""
    # --- File Paths & Model Names ---
    FAISS_INDEX_PATH = "./faiss_index_code_chunks"
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    GNN_EMBEDDINGS_PATH = "./z_dict.pt" # This is now the element_id->vector map

    # --- Reranking Hyperparameters ---
    TOP_K_SEMANTIC = 15  # How many initial candidates to retrieve
    ALPHA_BLEND = 0.5    # Blend factor (0.0 = pure structural, 1.0 = pure semantic)

    # --- Neo4j Connection (assumes env vars are set) ---
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("✅ Configuration class defined.")


# @title (FINAL ROBUST FIX) GNNReranker with Multi-Dimensional Handling
# Description: Handles heterogeneous graphs where node embeddings have different sizes (e.g. 128 vs 256).
# Fixes: "RuntimeError: stack expects each tensor to be equal size"

class GNNReranker:
    """
    A class to handle the entire GNN-based reranking process, from schema discovery
    to returning a final, blended list of results.
    """
    def __init__(self, config: RerankerConfig):
        self.config = config
        print("--- Initializing GNNReranker ---")
        self.driver = self._connect_neo4j()
        self.node_types = self._discover_node_types()
        self.name_to_elementId, self.path_to_elementId = self._build_lookup_maps()
        self.z_dict = self._load_gnn_embeddings()
        self.faiss_store = self._load_faiss_store()
        print("--- ✅ GNNReranker Initialized Successfully ---\n")

    def _connect_neo4j(self):
        try:
            driver = GraphDatabase.driver(self.config.NEO4J_URI, auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD))
            return driver
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return None

    def _run_query(self, cypher: str, params: dict = None) -> list[dict]:
        if not self.driver: return []
        with self.driver.session() as session:
            res = session.run(cypher, params or {})
            return [r.data() for r in res]

    def _discover_node_types(self) -> List[str]:
        exclude_labels = {'_Bloom_Perspective', '_Bloom_Scene', '_Graph', '_Entity'}
        records = self._run_query("CALL db.labels() YIELD label RETURN label")
        return sorted([rec['label'] for rec in records if rec['label'] not in exclude_labels])

    def _build_lookup_maps(self):
        name_to_elementId, path_to_elementId = {}, {}
        self.reverse_name_map = {}
        for ntype in self.node_types:
            records = self._run_query(f"MATCH (n:{ntype}) RETURN elementId(n) AS el_id, n.name AS name, n.path AS path")
            name_map, path_map = {}, {}
            for record in records:
                el_id = record['el_id']
                if record['name']:
                    name_key = record['name'].lower()
                    name_map[name_key] = el_id
                    self.reverse_name_map[el_id] = name_key
                if record['path']:
                    path_map[self._normalize_path(record['path'])] = el_id
            name_to_elementId[ntype] = name_map
            path_to_elementId[ntype] = path_map

        self.combined_name_map = {}
        for ntype, name_map in name_to_elementId.items():
            for name, el_id in name_map.items():
                self.combined_name_map[name] = (ntype, el_id)
        return name_to_elementId, path_to_elementId

    def _load_gnn_embeddings(self):
        try:
            z_dict = torch.load(self.config.GNN_EMBEDDINGS_PATH, map_location='cpu')
            return z_dict
        except FileNotFoundError:
            return None

    def _load_faiss_store(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
            return FAISS.load_local(self.config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            return None

    def _normalize_path(self, path: str) -> str:
        if not path: return ""
        return path.lower().replace("\\", "/").strip("./")

    def _normalize_filename_for_lookup(self, name: str) -> str:
        if not name: return ""
        name_norm = name.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
        return re.sub(r'_+', '_', name_norm).strip('_')

    def _get_ngrams(self, text: str, n: int) -> List[str]:
        words = re.findall(r'\w+', text)
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    def _link_query_entities(self, query: str) -> List[Tuple[str, str]]:
        query_norm = query.lower()
        backticked_entities = set(re.findall(r'`([^`]+)`', query))
        all_potential_entities = (
            backticked_entities.union(
            set(self._get_ngrams(query_norm, 3)),
            set(self._get_ngrams(query_norm, 2)),
            set(self._get_ngrams(query_norm, 1))
        ))
        linked_entities = []
        found_names = set()
        for entity_name in sorted(all_potential_entities, key=len, reverse=True):
            if entity_name in found_names: continue
            if entity_name in self.combined_name_map:
                ntype, element_id = self.combined_name_map[entity_name]
                linked_entities.append((ntype, element_id))
                for word in entity_name.split(): found_names.add(word)
                found_names.add(entity_name)
        return linked_entities

    # --- FIX: Return a dictionary of vectors, grouped by dimension ---
    def _get_query_vectors_by_dim(self, linked_entities: List[Tuple[str, str]]) -> Dict[int, torch.Tensor]:
        """
        Returns a dictionary of averaged query vectors, keyed by dimension.
        Example: {128: tensor(...), 256: tensor(...)}
        """
        if not linked_entities or not self.z_dict: return {}

        vectors_by_dim = {} # {128: [list of tensors], 256: [list of tensors]}

        for ntype, element_id in linked_entities:
            if ntype in self.z_dict and element_id in self.z_dict[ntype]:
                vec = self.z_dict[ntype][element_id]
                dim = vec.shape[0]

                if dim not in vectors_by_dim:
                    vectors_by_dim[dim] = []
                vectors_by_dim[dim].append(vec)

        # Average each group independently
        query_vectors_map = {}
        for dim, vec_list in vectors_by_dim.items():
            if vec_list:
                query_vectors_map[dim] = torch.stack(vec_list).mean(dim=0)

        return query_vectors_map

    def _map_chunk_to_kg_node_idx(self, chunk_metadata: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        rel_path = chunk_metadata.get("relative_path")
        if not rel_path: return None
        norm_path = self._normalize_path(rel_path)
        for ntype in self.node_types:
            if ntype in self.path_to_elementId and norm_path in self.path_to_elementId.get(ntype, {}):
                return (ntype, self.path_to_elementId[ntype][norm_path])
        filename = os.path.basename(norm_path)
        normalized_filename = self._normalize_filename_for_lookup(filename)
        for ntype in self.node_types:
            if ntype in self.name_to_elementId and normalized_filename in self.name_to_elementId.get(ntype, {}):
                 return (ntype, self.name_to_elementId[ntype][normalized_filename])
        return None

    # --- UPDATED RERANK METHOD ---
    def rerank(self, query: str, query_idx: int = -1, total_queries: int = -1) -> Optional[List[Dict[str, Any]]]:
        if not self.faiss_store or not self.z_dict:
            print("❌ Cannot perform reranking. FAISS store or GNN embeddings not loaded.")
            return None

        prefix = f"[Query {query_idx}/{total_queries}]" if query_idx > 0 else "[Query]"
        print(f"\n{prefix} [1/5] Performing semantic search for: '{query[:60]}...'")

        baseline_results = self.faiss_store.similarity_search_with_score(query, k=self.config.TOP_K_SEMANTIC)
        print(f"  -> Found {len(baseline_results)} initial candidates.")

        print(f"{prefix} [2/5] Linking query entities to knowledge graph...")
        linked_entities = self._link_query_entities(query)

        # FIX: Get map of vectors {128: vec, 256: vec}
        query_vectors_map = self._get_query_vectors_by_dim(linked_entities)

        if not query_vectors_map:
            print("  -> ⚠️ No entities found (or no embeddings). Reranking based on semantics alone.")
        else:
            print(f"  -> Found {len(linked_entities)} entities. Generated query vectors for dimensions: {list(query_vectors_map.keys())}")

        print(f"{prefix} [3/5] Calculating structural scores...")
        reranked_candidates = []

        for doc, semantic_score in baseline_results:
            candidate = {"doc": doc, "semantic_score": semantic_score, "graph_score": 0.0, "mapped_node": "N/A"}

            if query_vectors_map:
                mapping_result = self._map_chunk_to_kg_node_idx(doc.metadata)

                if mapping_result:
                    node_type, element_id = mapping_result
                    candidate["mapped_node"] = f"{node_type}:{self.reverse_name_map.get(element_id, 'N/A')}"
                    try:
                        node_embedding = self.z_dict[node_type][element_id]
                        target_dim = node_embedding.shape[0]

                        # --- FIX: Match Dimension Dynamically ---
                        if target_dim in query_vectors_map:
                            query_vec = query_vectors_map[target_dim]
                            candidate["graph_score"] = F.cosine_similarity(query_vec, node_embedding, dim=0).item()
                        else:
                            # If the query has entities, but NONE match this candidate's dimension
                            candidate["graph_score"] = 0.0

                    except (KeyError, IndexError):
                        pass

            reranked_candidates.append(candidate)

        print(f"{prefix} [4/5] Blending scores...")
        if not reranked_candidates: return []

        semantic_scores = [c['semantic_score'] for c in reranked_candidates]
        graph_scores = [c['graph_score'] for c in reranked_candidates]

        min_s, max_s = min(semantic_scores), max(semantic_scores)
        min_g, max_g = min(graph_scores), max(graph_scores)
        s_range = (max_s - min_s) if (max_s - min_s) != 0 else 1
        g_range = (max_g - min_g) if (max_g - min_g) != 0 else 1

        for c in reranked_candidates:
            norm_s = (c['semantic_score'] - min_s) / s_range
            norm_s_sim = 1.0 - norm_s
            norm_g = (c['graph_score'] - min_g) / g_range
            c['final_score'] = (self.config.ALPHA_BLEND * norm_s_sim) + ((1 - self.config.ALPHA_BLEND) * norm_g)

        reranked_results = sorted(reranked_candidates, key=lambda x: x['final_score'], reverse=True)
        return reranked_results

# =============================================================================
# EXAMPLE USAGE (A/B Test)
# =============================================================================
print("\n" + "="*40 + " RUNNING A/B TEST " + "="*40)
config = RerankerConfig()

# Check if all required components are available
if not all([config.NEO4J_URI, os.path.exists(config.FAISS_INDEX_PATH), os.path.exists(config.GNN_EMBEDDINGS_PATH)]):
    print("❌ ERROR: Missing one or more required components.")
    if not config.NEO4J_URI: print("  - NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD env vars not set.")
    if not os.path.exists(config.FAISS_INDEX_PATH): print(f"  - FAISS index not found at: {config.FAISS_INDEX_PATH}")
    if not os.path.exists(config.GNN_EMBEDDINGS_PATH): print(f"  - GNN embeddings not found at: {config.GNN_EMBEDDINGS_PATH}")
    print("Please run all preceding notebook cells to generate these artifacts.")
else:
    reranker = GNNReranker(config)

    # --- 2. Define Test Query and LLM ---
    # This is now a "natural" query, testing the fuzzy entity linking
    TEST_QUERY = "Technology Stack: The system uses `postgre_sql`, `cloud_sql`, and `spring_boot` as part of its technology stack, with dependencies on `prometheus` and `stackdriver` for monitoring."




    # TEST_QUERY = "The cartservice reads from and writes to the redis and alloy_db databases."



    if 'llm' not in globals():
        print("\nInitializing LLM for A/B test...")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        print("\nUsing existing LLM for A/B test...")

    reranked_results = reranker.rerank(TEST_QUERY)

    if reranked_results:
        baseline_results = sorted(reranked_results, key=lambda x: x['semantic_score'])

        def format_chunks_for_llm(docs: List[Dict[str, Any]], top_n=8) -> str:
            context_str = ""
            for i, doc in enumerate(docs[:top_n]):
                file_path = doc['doc'].metadata.get('relative_path', 'N/A')
                content = doc['doc'].page_content
                context_str += f"--- START CHUNK {i+1}: FILE: {file_path} ---\n"
                context_str += content
                context_str += f"\n--- END CHUNK {i+1} ---\n\n"
            return context_str

        baseline_context = format_chunks_for_llm(baseline_results)
        reranked_context = format_chunks_for_llm(reranked_results)

        # generation_prompt_template = ChatPromptTemplate.from_messages([
        #     ("system", """You are an expert microservice documentation author synthesizing a documentation section based on the provided context snippets. Your goal is to integrate all relevant details from the snippets into a cohesive, detailed, and factual answer to the user's specific query.

        #     **Instructions:**
        #     1.  **Synthesize, Don't Just Summarize:** Combine information from potentially multiple snippets to provide a comprehensive answer.
        #     2.  **Be Specific:** Extract and include specific function names, file names, dependencies, configurations, or architectural patterns mentioned in the context that directly address the query.
        #     3.  **Use All Relevant Context:** Make an effort to incorporate details from all provided snippets if they contribute to answering the query.
        #     4.  **Direct Answer:** Start the answer directly. Do not use phrases like "Based on the context provided...", "The context mentions...", or "According to the snippets...".
        #     """),
        #     ("human", """
        #     CONTEXT SNIPPETS:
        #     ---
        #     {context}
        #     ---

        #     QUERY / DOCUMENTATION TOPIC:
        #     {query}

        #     Synthesize a detailed documentation section addressing the query/topic using only the information available in the context snippets above:
        #     """)
        # ])

        generation_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert microservice documentation author. Your goal is to synthesize the provided context snippets into a highly **CONCISE, FACTUAL, and STRUCTURED** answer to the user's query.

            **Instructions for Conciseness:**
            1.  **Direct Answer:** Start the answer directly. Do not use conversational filler (e.g., 'Based on the context...', 'The service uses...').
            2.  **Preserve Key Facts & Dependencies (HIGH PRIORITY):** You MUST integrate all major architectural facts, dependencies, protocols, exposed ports, file references (if critical), and entity names (e.g., AlloyDB table names).
            3.  **Prioritize Lists & Tables (CONCISENESS):** Eliminate narrative prose. Convert lists of facts, components, configurations, and dependencies into high-density Markdown lists (dash/numbered) or tables where suitable.
            4.  **Use All Relevant Context:** Make an effort to incorporate details from all provided snippets if they contribute to answering the query.

            **Format Requirement:**
            Use Markdown headings (###) to separate the main topics (e.g., Redis Integration, AlloyDB Integration). Use bullet points for all detailed facts.
            """),
            ("human", """
            CONTEXT SNIPPETS:
            ---
            {context}
            ---

            QUERY / DOCUMENTATION TOPIC:
            {query}

            Synthesize a concise, structured documentation section addressing the query/topic using only the information available in the context snippets above:
            """)
        ])
        generation_chain = generation_prompt_template | llm | StrOutputParser()

        print("\n\n" + "="*40 + " LLM GENERATION A/B TEST " + "="*40)
        print("\n\n--- [A] Generating Documentation from BASELINE Context ---")
        baseline_answer = generation_chain.invoke({
            "context": baseline_context,
            "query": TEST_QUERY
        })
        print("\n--- LLM OUTPUT (BASELINE) ---")
        print(textwrap.fill(baseline_answer, width=100))

        print("\n\n--- [B] Generating Documentation from RERANKED Context ---")
        reranked_answer = generation_chain.invoke({
            "context": reranked_context,
            "query": TEST_QUERY
        })
        print("\n--- LLM OUTPUT (RERANKED) ---")
        print(textwrap.fill(reranked_answer, width=100))

        print("\n\n" + "="*30 + " BASELINE RESULTS (SEMANTIC ONLY) " + "="*30)
        baseline_df = pd.DataFrame({
            "Rank": [i+1 for i in range(len(baseline_results))],\
            "Semantic Score (Dist)": [f"{c['semantic_score']:.4f}" for c in baseline_results],\
            "Graph Score": [f"{c['graph_score']:.4f}" for c in baseline_results],\
            "Mapped Node": [c['mapped_node'] for c in baseline_results],\
            "File Path": [c['doc'].metadata.get('relative_path', 'N/A') for c in baseline_results]\
        })
        print(baseline_df.head(16).to_string())

        print("\n" + "="*28 + " RERANKED RESULTS (SEMANTIC + GRAPH) " + "="*28)
        reranked_df = pd.DataFrame({\
            "New Rank": [i+1 for i in range(len(reranked_results))],\
            "Final Score": [f"{c['final_score']:.4f}" for c in reranked_results],\
            "Graph Score": [f"{c['graph_score']:.4f}" for c in reranked_results],\
            "Semantic Score (Dist)": [f"{c['semantic_score']:.4f}" for c in reranked_results],\
            "Mapped Node": [c['mapped_node'] for c in reranked_results],\
            "File Path": [c['doc'].metadata.get('relative_path', 'N/A') for c in reranked_results]\
        })
        print(reranked_df.head(16).to_string())
        print("\n" + "="*86)

# @title (FINAL RECOMMENDED v2) Generate Queries from Outline (Parser + LLM Rephraser)
# Description: This cell implements a robust, two-stage query generation pipeline.
# 1. A fast parser extracts topics and ALL entities for a given bullet point.
# 2. An LLM chain rephrases these topics into single, complete, descriptive sentences
#    while preserving all backticked entities for the GNN.

import re
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


# This requires the 'llm' object to be initialized, e.g., from cell 'WyJRlU7OCD9M'
# Make sure this cell is run *after* the LLM is initialized.
if 'llm' not in globals():
    print("Initializing LLM for query generation...")
    # This assumes 'llm' is a global ChatOpenAI client.
    # If not, you must initialize it here.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
else:
    print("Using existing LLM for query generation.")


# Paste the full Markdown outline generated by the LLM in the previous step here.
DOCUMENTATION_OUTLINE = """
# 1. Introduction

The system architecture comprises a set of interconnected services designed to manage various aspects of a financial application. The top-level services include `accounts_db`, `balance_reader`, `contacts`, `frontend`, `ledger_db`, `ledgerwriter`, `transaction_history`, and `userservice`. Each service plays a distinct role in the system, contributing to the overall functionality and performance.

# 2. High-Level Architecture

The architecture is structured around several key services that interact through defined relationships. The `frontend` service connects to multiple services, including `userservice`, `balance_reader`, `contacts`, `transaction_history`, and `ledgerwriter`, facilitating user interactions and data flow. The `ledgerwriter` service connects to `balance_reader`, ensuring data consistency and integrity. The `userservice` publishes messages to the `jwt` message topic, enabling secure communication across the system.

# 3. Service Details

## `accounts_db`
- **Purpose:** Manages account-related data.
- **Main Responsibilities:** Handled by components such as `accounts_accounts_db`, `development_accounts_db`, and `production_accounts_db`.
- **Key Dependencies:** Depends on `postgre_sql` and `cloud_sql`.
- **Key Components:** Includes `accounts_db_config`, `initdb_accounts_db`, and `populate_accounts_db_accounts_db`.
- **Programming Language:** Not specified.

## `balance_reader`
- **Purpose:** Reads and provides balance information.
- **Main Responsibilities:** Managed by components like `balance_cache` and `balance_reader_controller`.
- **Key Dependencies:** Depends on `prometheus`, `zipkin`, and `stackdriver`.
- **Key Components:** Includes `balance_reader_controller`, `ledger_reader`, and `transaction_repository`.
- **Programming Language:** `java`

## `contacts`
- **Purpose:** Manages contact information.
- **Main Responsibilities:** Includes components such as `add_contact_contacts` and `contacts_db_contacts`.
- **Key Dependencies:** Depends on `prometheus`, `flask`, and `grpc`.
- **Key Components:** Includes `contacts_probe`, `contacts_table`, and `contacts_yaml`.
- **Programming Language:** `python`

## `frontend`
- **Purpose:** Acts as the user interface for the system.
- **Main Responsibilities:** Managed by components like `api_call_frontend` and `frontend_gateway`.
- **Key Dependencies:** Depends on `prometheus`, `flask`, and `grpc`.
- **Key Components:** Includes `frontend_ingress`, `frontend_yaml`, and `frontend_service_yaml`.
- **Programming Language:** `python`

## `ledger_db`
- **Purpose:** Manages ledger data.
- **Main Responsibilities:** Handled by components such as `ledger_reader` and `ledger_schema_config`.
- **Key Dependencies:** Depends on `postgre_sql` and `cloud_sql`.
- **Key Components:** Includes `ledger_db_config`, `ledger_reader_cache`, and `populate_ledger_db_ledger_db`.
- **Programming Language:** `java`

## `ledgerwriter`
- **Purpose:** Writes data to the ledger.
- **Main Responsibilities:** Managed by components like `ledger_writer_controller_ledgerwriter` and `transaction_validator_ledgerwriter`.
- **Key Dependencies:** Depends on `prometheus`, `guava`, and `jackson_databind`.
- **Key Components:** Includes `ledger_writer_yaml`, `ledgerwriter_probe`, and `transaction_repository_ledgerwriter`.
- **Programming Language:** `java`

## `transaction_history`
- **Purpose:** Manages transaction history data.
- **Main Responsibilities:** Includes components such as `transaction_history_controller` and `transaction_cache`.
- **Key Dependencies:** Depends on `prometheus`, `micrometer_registry_stackdriver`, and `spring_boot_starter_actuator`.
- **Key Components:** Includes `transactionhistory_probe`, `transaction_repository`, and `transaction_validator`.
- **Programming Language:** `java`

## `userservice`
- **Purpose:** Manages user-related operations.
- **Main Responsibilities:** Handled by components like `user_db_userservice` and `add_user_userservice`.
- **Key Dependencies:** Depends on `prometheus`, `flask`, and `grpc`.
- **Key Components:** Includes `userservice_probe`, `userservice_yaml`, and `jwt_key`.
- **Programming Language:** `python`

# 4. API & Communication View

- **API Gateway Pattern:** The `frontend_gateway` acts as an API Gateway, connecting to services like `userservice`, `balance_reader`, `contacts`, `transaction_history`, and `ledgerwriter`. It exposes endpoints such as `deposit`, `payment`, and `signup`.

- **Service Discovery:** The `prometheus` tool is a central service discovery mechanism, as many services depend on it for monitoring and metrics.

- **Direct Synchronous Communication:** The `frontend` service directly connects to `userservice`, `balance_reader`, `contacts`, `transaction_history`, and `ledgerwriter`.

- **Asynchronous Messaging:** The `jwt` message topic is published to by the `userservice`, facilitating secure token-based communication.

# 5. Data Architecture & Persistence View

- **`accounts_db`:** Writes to and reads from the `accounts` database.
- **`balance_reader`:** Reads from the `accounts_db` and `ledger_db` databases.
- **`contacts`:** Writes to and reads from the `contacts_db` database.
- **`ledger_db`:** Writes to and reads from the `ledger` and `transactions` databases.
- **`ledgerwriter`:** Writes to the `ledger_db` database.
- **`transaction_history`:** Writes to and reads from the `transactions` database.
- **`userservice`:** Writes to and reads from the `users` database.

# 6. Deployment & Infrastructure View

- **Service Definition:** Services like `ledgerwriter` and `transaction_history` are defined by configuration files such as `pom_xml` and `skaffold_yaml`.
- **Technology Stack:** The system uses `postgre_sql`, `cloud_sql`, and `spring_boot` as part of its technology stack, with dependencies on `prometheus` and `stackdriver` for monitoring.
- **Exposed Ports:** Services like `balance_reader`, `contacts`, and `ledgerwriter` expose port `8080`.

# 7. Cross-Cutting Concerns

- **Security:** The `jwt` message topic is a cross-cutting concern, providing token-based authentication across services.
- **Monitoring:** `prometheus` and `stackdriver` are used for monitoring and logging, ensuring system observability.

# 8. Technology Stack and Tools

- **`postgre_sql`:** Used as the primary database technology.
- **`cloud_sql`:** Provides cloud-based database management.
- **`prometheus`:** Used for monitoring and metrics collection.
- **`stackdriver`:** Provides logging and monitoring capabilities.
- **`flask`, `grpc`, `gunicorn`:** Used in the `frontend` and `userservice` for web and API services.
- **`spring_boot`:** Used in `ledgerwriter` and `transaction_history` for building Java applications.
- **`maven`:** Used for project management and build automation in Java services.
"""

# --- STEP 1: The "Unnatural" Parser (v4 - CORRECTED) ---
def parse_outline_to_topics(outline_text: str) -> List[str]:
    """
    (Internal Step)
    Parses the Markdown outline into a list of "topic: entity_list" strings,
    injecting the main Section Title context for non-Service-specific points.
    """
    topics = []
    current_service = None
    current_section = None # Tracks the current Level 1 Section Title

    lines = outline_text.strip().split('\n')
    bullet_regex = re.compile(r'^\s*-\s*\*\*(.*?):\*\*\s*(.*)')

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. Update Section Tracking (Level 1 Header: # Section Title)
        if line.startswith('# '):
            # Capture the Section Title, stripping the markdown and numbering
            current_section = re.sub(r'^\#\s*(\d+\.?\s*)?', '', line).strip()
            current_service = None # Reset service when main section changes
            # Skip the Intro/Architecture sections themselves
            if any(key in current_section for key in ["Introduction", "High-Level Architecture"]):
                 continue

        # 2. Update Service Tracking (Level 2 Header: ## `service_name`)
        service_match = re.match(r'^## `(.+?)`$', line)
        if service_match:
            current_service = f"`{service_match.group(1)}`"
            # General service overview query (no section prefix needed here)
            topics.append(f"High-level overview and purpose of the {current_service} service.")
            continue

        # 3. Process Bullet Points and Prose
        bullet_match = bullet_regex.match(line)
        if bullet_match:
            label = bullet_match.group(1).strip().lower()
            content = bullet_match.group(2).strip().rstrip('.')

            # --- CONTEXT INJECTION LOGIC ---
            if current_service:
                # Inside Service Details: Use service name as prefix
                topic_prefix = f"{current_service} {label}: "
            elif current_section:
                # Inside General Section (e.g., Data Architecture): Use Section Title as prefix
                topic_prefix = f"[Section: {current_section}] {label}: "
            else:
                # Fallback for generic lines (rare but safe)
                topic_prefix = f"{label}: "

            # ... (Extraction logic remains the same)
            if re.findall(r'`([^`]+)`', content):
                # Entity list case
                entities = re.findall(r'`([^`]+)`', content)
                all_entities_content = ", ".join([f"`{e}`" for e in entities])
                topics.append(f"{topic_prefix} entities: {all_entities_content}.")
            else:
                # Prose case (Purpose, Language)
                topics.append(f"{topic_prefix} {content}")

        # 4. Handle other lines (e.g., introduction prose)
        elif not line.startswith('#') and '`' in line:
            # Inject section context for high-level prose lines (like Query 4-7)
            context_prefix = f"[{current_section}] " if current_section and not current_service else ""
            topics.append(context_prefix + line)

    return topics

# --- STEP 2: The LLM Rephrasing Chain (Unchanged) ---

# Define Pydantic model for the expected LLM output
class GeneratedQueryList(BaseModel):
    queries: List[str] = Field(description="A list of descriptive, natural language sentences.")

# Create the parser
rephrasing_parser = PydanticOutputParser(pydantic_object=GeneratedQueryList)

# Define the prompt template for the LLM
rephrasing_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert technical writer. Your task is to convert the following
list of "topic-style" notes into a list of full, descriptive sentences.

**CRITICAL RULES:**
1.  **PRESERVE BACKTICKS:** You MUST preserve all backticked entity names
    (e.g., `adservice`, `g_rpc`) exactly as they appear.
2.  **CREATE STANDALONE SENTENCES:** Ensure every sentence is semantically complete.
3.  **RESOLVE CONTEXT/PRONOUNS:** If a topic contains a `[Section: X]` prefix, or if a sentence starts with 'It' or 'They', integrate the specific section title or entity name into the resulting sentence to ensure it is fully self-contained. For example, turn '[Section: Deployment View] Services expose ports' into 'In the Deployment View, services expose ports...'.
4.  **FORMALIZE CONTEXT PREFIXES:** If a sentence begins with a section name (e.g., 'In the Technology Stack and Tools'), **append the word 'section' or 'view'** to the section name for formality and natural flow.
5.  **DO NOT ADD NEW INFO:** Do not invent any information not present in the notes.
6.  **BE NATURAL:** Make the sentences sound like they were written by a human.

**EXAMPLES:**
- **Input Note:** `adservice` purpose: Manages advertisements.
- **Good Output:** "The `adservice` service manages advertisements."

- **Input Note:** `adservice` key dependencies: `g_rpc`, `jackson`, `log4j`.
- **Good Output:** "The `adservice` has key dependencies on `g_rpc`, `jackson`, and `log4j`."

- **Input Note:** `adservice` main responsibilities: `ad_adservice`, `ad_service_client_adservice`.
- **Good Output:** "The main responsibilities of the `adservice` are handled by components like `ad_adservice` and `ad_service_client_adservice`."

- **Input Note:** API Gateway Pattern: The `frontend_gateway` acts as an API Gateway...
- **Good Output:** "For the API Gateway Pattern, the `frontend_gateway` acts as an API Gateway, exposing the `frontend_ingress` endpoint and connecting to the `frontend` service. It also exposes the `frontend_route` API endpoint."

**Here is the list of notes to rephrase:**
{topic_list}

{format_instructions}
"""
)

# # Build the rephrasing chain
# query_rephrasing_chain = rephrasing_prompt_template | llm | rephrasing_parser

# # --- Main Execution ---
# print("--- STAGE 1: Parsing outline into topic-entity notes ---")
# intermediate_topics = parse_outline_to_topics(DOCUMENTATION_OUTLINE)
# print(f"✅ Parsed {len(intermediate_topics)} topics.")

# # Join the list of topics into a single string for the LLM prompt
# intermediate_topics_str = "\n".join(f"- {t}" for t in intermediate_topics)

# print("\n--- STAGE 2: Sending topics to LLM for rephrasing ---")
# generated_query_obj = query_rephrasing_chain.invoke({
#     "topic_list": intermediate_topics_str,
#     "format_instructions": rephrasing_parser.get_format_instructions()
# })
# generated_queries = generated_query_obj.queries
# print(f"✅ Successfully generated {len(generated_queries)} rephrased queries.\n")


# # --- Display Results ---
# print("--- Sample of *FINAL* Generated Queries (Descriptive Sentences) ---")
# for i, query in enumerate(generated_queries): # Print the first 15 as a sample
#     print(f"Query {i+1}: {query}")

# @title 4.0 Generate Full Documentation (Experiment Runner) [FINAL - Cost Aware + Logs]
import time
import re
import os
from tqdm.notebook import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback # For Token Counting

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
TOP_K_CONTEXT = 5
ALPHA_PARAM = 0.5
OUTPUT_DIR = "./generated_docs"
EXTRACTED_ARTIFACTS_DIR = "./extracted_artifacts"

# Apply Config
if 'reranker' in globals():
    reranker.config.ALPHA_BLEND = ALPHA_PARAM
    print(f"✅ EXPERIMENT SETUP: Alpha={ALPHA_PARAM} | Top-K={TOP_K_CONTEXT}")
else:
    print("⚠️ Warning: 'reranker' not found. Run Cell 55 first.")

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Buffers
full_doc_baseline = []
full_doc_reranked = []

# Metrics
metrics = {
    "rag": {"time": 0.0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
    "gnn":      {"time": 0.0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
}

# ==========================================
# 2. HELPER: FORMAT DOCS
# ==========================================
def format_docs(docs, top_n, max_chars=50000):
    context_str = ""
    added_files = set()

    for i, doc_data in enumerate(docs[:top_n]):
        if isinstance(doc_data, dict): doc = doc_data['doc']
        else: doc = doc_data

        rel_path = doc.metadata.get('relative_path', 'N/A')
        if rel_path in added_files: continue

        full_file_path = os.path.join(EXTRACTED_ARTIFACTS_DIR, rel_path)
        content_source = "CHUNK_ONLY"
        file_content = doc.page_content

        if os.path.exists(full_file_path):
            try:
                with open(full_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()
                content_source = "FULL_FILE_FROM_DISK"
            except: pass

        if len(file_content) > max_chars:
            file_content = file_content[:max_chars] + f"\n...[TRUNCATED]..."

        context_str += f"--- SOURCE: {rel_path} ({content_source}) ---\n{file_content}\n\n"
        added_files.add(rel_path)
    return context_str

# ==========================================
# 3. ROBUST PLAN GENERATOR (The Logic You Asked About)
# ==========================================
# This function reads the 'DOCUMENTATION_OUTLINE' variable directly.
def create_execution_plan(outline_text):
    plan = []
    lines = outline_text.strip().split('\n')
    bullet_regex = re.compile(r'^\s*-\s*\*\*(.*?):\*\*\s*(.*)')

    inferred_services = INFERRED_TOP_LEVEL_SERVICES if 'INFERRED_TOP_LEVEL_SERVICES' in globals() else []

    def normalize_service_key(name: str) -> str:
        key = name.lower().strip('`* ')
        key = key.replace('-', '_')
        for suffix in ['_service', '-service', 'service']:
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key.strip('_-')

    def is_known_service(name: str) -> bool:
        if not inferred_services:
            return True
        return normalize_service_key(name) in {normalize_service_key(s) for s in inferred_services}

    def get_service_display_name(canonical_name: str) -> str:
        if 'IDENTIFIED_SERVICES_DETAILS' in globals() and IDENTIFIED_SERVICES_DETAILS:
            for s_info in IDENTIFIED_SERVICES_DETAILS:
                if normalize_service_key(s_info.name) == normalize_service_key(canonical_name):
                    return s_info.name
        return canonical_name

    # Context Trackers
    current_section = None
    current_service = None
    included_services = set()
    service_details_header_seen = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Case 1: Main Section Headers (e.g., "# 3. Service Details")
        if line.startswith('# '):
            current_section = re.sub(r'^#\s*(\d+\.?\s*)?', '', line).strip()
            current_service = None  # Reset service context
            plan.append({'type': 'header', 'text': line})
            if 'service details' in (current_section or '').lower():
                service_details_header_seen = True
            continue

        # Case 2: Service Sub-Headers (e.g., "## `cartservice`")
        if line.startswith('##'):
            clean_header = re.sub(r'^##\s*', '', line).strip(' `*')
            if clean_header:
                if inferred_services and not is_known_service(clean_header):
                    continue
                current_service = f"`{clean_header}`"
                included_services.add(normalize_service_key(clean_header))
                plan.append({'type': 'header', 'text': line})
                # Add an implicit "Overview" query for the service
                plan.append({
                    'type': 'query',
                    'text': f"High-level overview and purpose of the {current_service} service.",
                    'section_context': current_section  # Track for logging
                })
                continue

        # Case 3: Bullet Points (The Queries)
        query_text = None
        bullet_match = bullet_regex.match(line)

        if bullet_match:
            label = bullet_match.group(1).strip()
            content = bullet_match.group(2).strip().rstrip('.')
            label_key = normalize_service_key(label)

            # If we're in Service Details and this bullet is a service name, handle it explicitly
            if (current_section or '').lower().startswith('service details') and not current_service:
                if inferred_services and not is_known_service(label):
                    continue
                if inferred_services:
                    display = get_service_display_name(label)
                    current_service = f"`{display}`"
                    included_services.add(normalize_service_key(display))
                    plan.append({'type': 'header', 'text': f"## {current_service}"})
                    query_text = f"{current_service} purpose: {content}"
                else:
                    prefix = f"{label}: "
                    query_text = f"{prefix} {content}"
            else:
                # Construct Query: "[Service Name] [Label]: [Content]"
                prefix = f"{current_service} {label_key}: " if current_service else f"{label_key}: "
                query_text = f"{prefix} {content}"

        elif line.startswith('- ') or line.startswith('* '):
            clean_text = line.lstrip('-* ').strip()
            query_text = f"{current_service}: {clean_text}" if current_service else clean_text

        elif not line.startswith('#') and len(line) > 10:
            prefix = f"[{current_section}] " if current_section else ""
            query_text = prefix + line

        if query_text:
            plan.append({
                'type': 'query',
                'text': query_text,
                'section_context': current_section or "General"
            })

    # Add missing service sections based on inferred services
    if inferred_services:
        missing = [s for s in inferred_services if normalize_service_key(s) not in included_services]
        if missing:
            if not service_details_header_seen:
                plan.append({'type': 'header', 'text': '# Service Details'})
            for svc in missing:
                display = get_service_display_name(svc)
                svc_ref = f"`{display}`"
                plan.append({'type': 'header', 'text': f"## {svc_ref}"})
                plan.append({'type': 'query', 'text': f"High-level overview and purpose of the {svc_ref} service."})
                plan.append({'type': 'query', 'text': f"{svc_ref} main responsibilities: "})
                plan.append({'type': 'query', 'text': f"{svc_ref} key dependencies: "})
                plan.append({'type': 'query', 'text': f"{svc_ref} key components: "})
                plan.append({'type': 'query', 'text': f"{svc_ref} programming language: "})

    return plan

# Execute the Plan Generator
if 'DOCUMENTATION_OUTLINE' not in globals():
    print("❌ ERROR: 'DOCUMENTATION_OUTLINE' not found. Please run the Outline Generation cell first.")
else:
    execution_plan = create_execution_plan(DOCUMENTATION_OUTLINE)
    print(f"📋 Execution Plan Parsed: {len(execution_plan)} steps.")


# ==========================================
# 4. GENERATION LOOP
# ==========================================
def get_query_intent_instruction(query_text):
    q = query_text.lower()
    if "introduction" in q: return "Write a high-level executive summary."
    elif "architecture" in q: return "Focus on service relationships and data flow."
    elif "details" in q: return "Focus strictly on technical details (ports, libs)."
    else: return "Be concise and factual."

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict technical writer.
    **CRITICAL TRUTH RULE:**
    You must verify every claim against the provided **SOURCE CODE CONTEXT**.
    - If the code doesn't show it, DO NOT mention it.
    - Ignore metadata if it contradicts the file content.

    **Format:** No preamble. Dense bullet points.
    """),
    ("human", "CONTEXT:\n{context}\n\nQUERY: {query}\n\nintent: {intent}")
])
generation_chain = generation_prompt | llm | StrOutputParser()

query_counter = 0
total_queries = len([x for x in execution_plan if x['type'] == 'query'])

# LOGGING TRACKER
current_display_section = "Start"

print("\n🚀 STARTING GENERATION LOOP...")

for step in tqdm(execution_plan, desc="Generating"):
    # 1. HEADER: Just append to doc and update log tracker
    if step['type'] == 'header':
        current_display_section = step['text'].strip()
        full_doc_baseline.append(f"\n{step['text']}\n")
        full_doc_reranked.append(f"\n{step['text']}\n")
        continue

    # 2. QUERY: The actual work
    query = step['text']
    query_counter += 1

    # --- INSIGHT LOGGING ---
    # This shows you exactly what text is driving the retrieval
    print(f"🔎 [Ctx: {current_display_section[:20]}...] Query: \"{query}\"")
    # -----------------------

    # Retrieval
    reranked_pool = reranker.rerank(query, query_idx=query_counter, total_queries=total_queries)
    if not reranked_pool:
        print(f"   ⚠️ No context found.")
        continue

    # Sorting
    baseline_pool = sorted(reranked_pool, key=lambda x: x['semantic_score'], reverse=True)

    # Context Formatting
    context_base = format_docs(baseline_pool, TOP_K_CONTEXT)
    context_gnn = format_docs(reranked_pool, TOP_K_CONTEXT)

    intent = get_query_intent_instruction(query)

    # --- GENERATE BASELINE ---
    try:
        start_base = time.time()
        with get_openai_callback() as cb_base:
            ans_base = generation_chain.invoke({"context": context_base, "query": query, "intent": intent})

            metrics["rag"]["time"] += (time.time() - start_base)
            metrics["rag"]["total_tokens"] += cb_base.total_tokens
            metrics["rag"]["prompt_tokens"] += cb_base.prompt_tokens
            metrics["rag"]["completion_tokens"] += cb_base.completion_tokens
            metrics["rag"]["cost_usd"] += cb_base.total_cost
        full_doc_baseline.append(f"\n{ans_base}\n")
    except Exception as e: print(f"Base Error: {e}")

    # --- GENERATE GNN ---
    try:
        start_gnn = time.time()
        with get_openai_callback() as cb_gnn:
            ans_gnn = generation_chain.invoke({"context": context_gnn, "query": query, "intent": intent})

            metrics["gnn"]["time"] += (time.time() - start_gnn)
            metrics["gnn"]["total_tokens"] += cb_gnn.total_tokens
            metrics["gnn"]["prompt_tokens"] += cb_gnn.prompt_tokens
            metrics["gnn"]["completion_tokens"] += cb_gnn.completion_tokens
            metrics["gnn"]["cost_usd"] += cb_gnn.total_cost
        full_doc_reranked.append(f"\n{ans_gnn}\n")
    except Exception as e: print(f"GNN Error: {e}")

    time.sleep(0.01)

# Save
suffix = f"k{TOP_K_CONTEXT}_a{ALPHA_PARAM}"
with open(os.path.join(OUTPUT_DIR, f"documentation_rag_{suffix}.md"), "w") as f: f.write("".join(full_doc_baseline))
with open(os.path.join(OUTPUT_DIR, f"documentation_gnn_{suffix}.md"), "w") as f: f.write("".join(full_doc_reranked))

# Report
print("\n" + "="*50)
print(f"📊 COST REPORT (Top-K={TOP_K_CONTEXT})")
print("="*50)
print(f"🔹 RAG: ${metrics['rag']['cost_usd']:.4f} ({metrics['rag']['time']:.1f}s)")
print(f"🔸 GNN:      ${metrics['gnn']['cost_usd']:.4f} ({metrics['gnn']['time']:.1f}s)")
print("="*50)

import time
import re
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.callbacks import get_openai_callback # For Token Counting

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
TOP_K_CONTEXT = 5           # Final number of chunks to send to LLM
KG_HOPS_BROAD = 2           # For Intro/Architecture (Explore Neighborhood)
KG_HOPS_STRICT = 1          # For Details/Infra (Direct Connections Only)
OUTPUT_DIR = "./generated_docs"
EXTRACTED_ARTIFACTS_DIR = "./extracted_artifacts"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Buffers
full_doc_kg_ablation = []

# Metrics Tracking
metrics = {
    "kg_ablation": {"time": 0.0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
}

# ==========================================
# 2. GRAPH-FIRST RETRIEVER CLASS
# ==========================================
class GraphFirstRetriever:
    def __init__(self, original_reranker):
        self.driver = original_reranker.driver
        self.faiss_store = original_reranker.faiss_store
        # Access the embeddings model wrapper directly from the store
        self.embeddings_model = self.faiss_store.embeddings

        # Reuse helpers
        self._link_query_entities = original_reranker._link_query_entities

        # Build Reverse Index (Path -> [Document])
        print("   [KG-First] Building Path->Chunk Reverse Index...")
        self.path_to_docs = {}
        # Reuse the normalization function from reranker
        self._normalize_path = original_reranker._normalize_path
        # Access the underlying docstore from FAISS
        for doc_id, doc in self.faiss_store.docstore._dict.items():
            path = doc.metadata.get('relative_path')
            if path:
                # Normalize path to match Neo4j path format
                norm_path = self._normalize_path(path)
                if norm_path:
                    if norm_path not in self.path_to_docs:
                        self.path_to_docs[norm_path] = []
                    self.path_to_docs[norm_path].append(doc)
        print(f"   [KG-First] Index ready. Mapped {len(self.path_to_docs)} files.")

    def _get_cypher_strategy(self, query_text: str) -> tuple[int, list[str]]:
        """
        Determines Hops and Edge Types based on the query context.
        VALIDATED TYPES: CONNECTS_TO, EXPOSES_ENDPOINT, EXPOSES_PORT, PUBLISHES_TO,
                         SUBSCRIBES_TO, WRITES_TO, READS_FROM, HAS_COMPONENT,
                         DEPENDS_ON, WRITTEN_IN, DEFINED_BY
        """
        q_lower = query_text.lower()

        # 1. Introduction / High-Level Architecture
        if any(x in q_lower for x in ["overview", "architecture", "introduction", "high-level"]):
            return KG_HOPS_BROAD, ["CONNECTS_TO", "DEPENDS_ON", "PUBLISHES_TO", "SUBSCRIBES_TO", "HAS_COMPONENT"]

        # 2. Service Details (The deep dive)
        elif "service details" in q_lower or "responsibilities" in q_lower:
            return KG_HOPS_STRICT, ["HAS_COMPONENT", "WRITTEN_IN", "DEFINED_BY", "CONNECTS_TO", "EXPOSES_PORT"]

        # 3. API & Communication
        elif "api" in q_lower or "communication" in q_lower or "gateway" in q_lower:
            return KG_HOPS_BROAD, ["EXPOSES_ENDPOINT", "EXPOSES_PORT", "CONNECTS_TO", "PUBLISHES_TO", "SUBSCRIBES_TO"]

        # 4. Data Architecture
        elif "data" in q_lower or "persistence" in q_lower or "database" in q_lower:
            return KG_HOPS_BROAD, ["WRITES_TO", "READS_FROM", "CONNECTS_TO"]

        # 5. Infrastructure & Deployment
        elif any(x in q_lower for x in ["deployment", "infrastructure", "configuration", "technology", "stack", "tools"]):
            return KG_HOPS_STRICT, ["DEFINED_BY", "DEPENDS_ON", "WRITTEN_IN"]

        # 6. Cross-Cutting Concerns
        elif "cross-cutting" in q_lower or "security" in q_lower or "logging" in q_lower:
            return KG_HOPS_BROAD, ["DEPENDS_ON", "HAS_COMPONENT"]

        # Default / Fallback - Use broader strategy when no specific pattern matches
        return KG_HOPS_BROAD, ["CONNECTS_TO", "DEPENDS_ON", "HAS_COMPONENT", "WRITES_TO", "READS_FROM"]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # 1. Entity Extraction
        linked_entities = self._link_query_entities(query)
        if not linked_entities: return []

        # 2. Prepare Cypher Query
        hops, edge_types = self._get_cypher_strategy(query)
        edge_clause = ""
        if edge_types:
            types_str = "|".join([f"`{t}`" for t in edge_types])
            edge_clause = f":{types_str}"

        rels_str = f"[{edge_clause}*1..{hops}]"
        q_ids = [eid for _, eid in linked_entities]

        # Cypher: Find all nodes reachable from Query Entities
        cypher = f"""
        MATCH (s)
        WHERE elementId(s) IN $q_ids
        MATCH (s)-{rels_str}-(t)
        WHERE t.path IS NOT NULL
        RETURN DISTINCT t.path as file_path
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher, q_ids=q_ids)
                target_paths = [record["file_path"] for record in result]
        except Exception as e:
            print(f"   [KG-First] Cypher Error: {e}")
            target_paths = []

        # Fallback: If no paths found and edge_types were restrictive, try with broader edge types
        if not target_paths and edge_types:

            # Retry with broader edge types and more hops
            fallback_edge_types = ["CONNECTS_TO", "DEPENDS_ON", "HAS_COMPONENT", "WRITES_TO", "READS_FROM", "PUBLISHES_TO", "SUBSCRIBES_TO"]
            fallback_hops = max(hops, KG_HOPS_BROAD)
            types_str = "|".join([f"`{t}`" for t in fallback_edge_types])
            fallback_rels_str = f"[:{types_str}*1..{fallback_hops}]"

            fallback_cypher = f"""
            MATCH (s)
            WHERE elementId(s) IN $q_ids
            MATCH (s)-{fallback_rels_str}-(t)
            WHERE t.path IS NOT NULL
            RETURN DISTINCT t.path as file_path
            """

            try:
                with self.driver.session() as session:
                    result = session.run(fallback_cypher, q_ids=q_ids)
                    target_paths = [record["file_path"] for record in result]

            except Exception as e:
                target_paths = []

        if not target_paths:
            # Removed log_debug since it's commented out earlier
            # log_debug("FAILURE: no target paths after fallback", {"q_ids_count": len(q_ids), "hops": hops, "edge_types": edge_types})
            # Fallback: semantic retrieval when KG yields no paths
            try:
                semantic_hits = self.faiss_store.similarity_search_with_score(query, k=top_k)
                # Removed log_debug since it's commented out earlier
                # log_debug("fallback semantic search used (no target paths)", {"hit_count": len(semantic_hits)})
                return [
                    {"doc": doc, "semantic_score": float(score), "final_score": float(score)}
                    for doc, score in semantic_hits
                ]
            except Exception as e:
                # Removed log_debug since it's commented out earlier
                # log_debug("fallback semantic search failed (no target paths)", {"error": str(e)[:200]})
                return []

        # 3. Fetch Chunks (The Candidate Set)
        candidate_docs = []
        paths_found = 0
        paths_not_found = []
        for path in target_paths:
            if not path:
                continue
            # Normalize the path from Neo4j to match path_to_docs keys
            norm_path = self._normalize_path(path)
            # Try normalized path first, then original path as fallback
            matched = False
            if norm_path and norm_path in self.path_to_docs:
                candidate_docs.extend(self.path_to_docs[norm_path])
                paths_found += 1
                matched = True
            elif path in self.path_to_docs:
                # Fallback: try original path (in case normalization changed it unexpectedly)
                candidate_docs.extend(self.path_to_docs[path])
                paths_found += 1
                matched = True

            if not matched:
                paths_not_found.append(path)

        if not candidate_docs:
            # Removed log_debug since it's commented out earlier
            # log_debug("FAILURE: no candidate docs", {"target_paths_count": len(target_paths), "paths_found": paths_found, "sample_not_found": paths_not_found[:3]})
            # Fallback: semantic retrieval when paths map to no docs
            try:
                semantic_hits = self.faiss_store.similarity_search_with_score(query, k=top_k)
                # Removed log_debug since it's commented out earlier
                # log_debug("fallback semantic search used (no candidate docs)", {"hit_count": len(semantic_hits)})
                return [
                    {"doc": doc, "semantic_score": float(score), "final_score": float(score)}
                    for doc, score in semantic_hits
                ]
            except Exception as e:
                # Removed log_debug since it's commented out earlier
                # log_debug("fallback semantic search failed (no candidate docs)", {"error": str(e)[:200]})
                return []

        # 4. Semantic Sorting (The "Ranking")
        query_embedding = self.embeddings_model.embed_query(query)
        chunk_texts = [d.page_content for d in candidate_docs]
        chunk_embeddings = self.embeddings_model.embed_documents(chunk_texts)

        sim_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]

        scored_candidates = []
        for i, doc in enumerate(candidate_docs):
            scored_candidates.append({
                "doc": doc,
                "semantic_score": float(sim_scores[i]),
                "final_score": float(sim_scores[i])
            })

        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_candidates[:top_k]

# Initialize (requires 'reranker' from Cell 55)
if 'reranker' in globals():
    kg_retriever = GraphFirstRetriever(reranker)
    print("✅ Graph-First Retriever Initialized.")
else:
    print("❌ Error: 'reranker' object not found. Please run Cell 55 first.")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def format_docs_kg(docs, top_n, max_chars=50000):
    context_str = ""
    added_files = set()
    for i, doc_data in enumerate(docs[:top_n]):
        doc = doc_data['doc']
        rel_path = doc.metadata.get('relative_path', 'N/A')
        if rel_path in added_files: continue

        full_file_path = os.path.join(EXTRACTED_ARTIFACTS_DIR, rel_path)
        content_source = "CHUNK_ONLY"
        file_content = doc.page_content

        if os.path.exists(full_file_path):
            try:
                with open(full_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()
                content_source = "FULL_FILE_FROM_DISK"
            except: pass

        if len(file_content) > max_chars:
            file_content = file_content[:max_chars] + f"\n...[TRUNCATED {len(file_content)-max_chars} chars]..."

        context_str += f"--- SOURCE: {rel_path} ({content_source}) ---\n{file_content}\n\n"
        added_files.add(rel_path)
    return context_str

def get_query_intent_instruction(query_text):
    q = query_text.lower()
    if "introduction" in q: return "Write a high-level executive summary."
    elif "architecture" in q: return "Focus on service relationships and data flow."
    elif "details" in q: return "Focus strictly on technical details (ports, libs)."
    else: return "Be concise and factual."

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict technical writer.

    **CRITICAL TRUTH RULE:**
    You must verify every claim against the provided **SOURCE CODE CONTEXT**.
    - The retrieval system (Knowledge Graph) may sometimes provide outdated metadata or "ghost" nodes (e.g. referencing signal handlers or variables that no longer exist in the file).
    - **IF** the provided source code text does not explicitly show the logic, **DO NOT** mention it.
    - Ignore any "evidence" or "metadata" that contradicts the actual file content.
    - Only write what you can see in the file content.

    **Format:**
    - No preamble.
    - Use dense bullet points.
    - List specific versions/ports ONLY if visible in the text.
    """),
    ("human", "CONTEXT:\n{context}\n\nQUERY: {query}\n\nintent: {intent}")
])
generation_chain = generation_prompt | llm | StrOutputParser()

# ==========================================
# 4. ROBUST PLAN GENERATOR (Defined locally to be independent)
# ==========================================
def create_execution_plan_local(outline_text):
    """
    Local version of the plan generator to ensure this cell can run
    independently of Cell 56.
    """
    plan = []
    lines = outline_text.strip().split('\n')
    bullet_regex = re.compile(r'^\s*-\s*\*\*(.*?):\*\*\s*(.*)')

    inferred_services = INFERRED_TOP_LEVEL_SERVICES if 'INFERRED_TOP_LEVEL_SERVICES' in globals() else []

    def normalize_service_key(name: str) -> str:
        key = name.lower().strip('`* ')
        key = key.replace('-', '_')
        for suffix in ['_service', '-service', 'service']:
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key.strip('_-')

    def is_known_service(name: str) -> bool:
        if not inferred_services:
            return True
        return normalize_service_key(name) in {normalize_service_key(s) for s in inferred_services}

    def get_service_display_name(canonical_name: str) -> str:
        if 'IDENTIFIED_SERVICES_DETAILS' in globals() and IDENTIFIED_SERVICES_DETAILS:
            for s_info in IDENTIFIED_SERVICES_DETAILS:
                if normalize_service_key(s_info.name) == normalize_service_key(canonical_name):
                    return s_info.name
        return canonical_name

    current_section = None
    current_service = None
    included_services = set()
    service_details_header_seen = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Headers
        if line.startswith('# '):
            current_section = re.sub(r'^#\s*(\d+\.?\s*)?', '', line).strip()
            current_service = None
            plan.append({'type': 'header', 'text': line})
            if 'service details' in (current_section or '').lower():
                service_details_header_seen = True
            continue

        # Service Headers
        if line.startswith('##'):
            clean_header = re.sub(r'^##\s*', '', line).strip(' `*')
            if clean_header:
                if inferred_services and not is_known_service(clean_header):
                    continue
                current_service = f"`{clean_header}`"
                included_services.add(normalize_service_key(clean_header))
                plan.append({'type': 'header', 'text': line})
                plan.append({'type': 'query', 'text': f"High-level overview and purpose of the {current_service} service."})
                continue

        # Queries
        query_text = None
        bullet_match = bullet_regex.match(line)

        if bullet_match:
            label = bullet_match.group(1).strip()
            content = bullet_match.group(2).strip().rstrip('.')
            label_key = normalize_service_key(label)

            if (current_section or '').lower().startswith('service details') and not current_service:
                if inferred_services and not is_known_service(label):
                    continue
                if inferred_services:
                    display = get_service_display_name(label)
                    current_service = f"`{display}`"
                    included_services.add(normalize_service_key(display))
                    plan.append({'type': 'header', 'text': f"## {current_service}"})
                    query_text = f"{current_service} purpose: {content}"
                else:
                    prefix = f"{label}: "
                    query_text = f"{prefix} {content}"
            else:
                prefix = f"{current_service} {label_key}: " if current_service else f"{label_key}: "
                query_text = f"{prefix} {content}"
        elif line.startswith('- ') or line.startswith('* '):
            clean_text = line.lstrip('-* ').strip()
            query_text = f"{current_service}: {clean_text}" if current_service else clean_text
        elif not line.startswith('#') and len(line) > 10:
             prefix = f"[{current_section}] " if current_section else ""
             query_text = prefix + line

        if query_text:
            plan.append({'type': 'query', 'text': query_text})

    # Add missing service sections based on inferred services
    if inferred_services:
        missing = [s for s in inferred_services if normalize_service_key(s) not in included_services]
        if missing:
            if not service_details_header_seen:
                plan.append({'type': 'header', 'text': '# Service Details'})
            for svc in missing:
                display = get_service_display_name(svc)
                svc_ref = f"`{display}`"
                plan.append({'type': 'header', 'text': f"## {svc_ref}"})
                plan.append({'type': 'query', 'text': f"High-level overview and purpose of the {svc_ref} service."})
                plan.append({'type': 'query', 'text': f"{svc_ref} main responsibilities: "})
                plan.append({'type': 'query', 'text': f"{svc_ref} key dependencies: "})
                plan.append({'type': 'query', 'text': f"{svc_ref} key components: "})
                plan.append({'type': 'query', 'text': f"{svc_ref} programming language: "})

    return plan

# ==========================================
# 5. EXECUTION LOOP (KG Ablation Only) - Cost Aware
# ==========================================
if 'DOCUMENTATION_OUTLINE' not in globals():
    print("❌ Critical Error: 'DOCUMENTATION_OUTLINE' variable not found. Please run the Outline Generation cell (Cell 8/19) first.")
else:
    # Generate the plan locally
    execution_plan = create_execution_plan_local(DOCUMENTATION_OUTLINE)
    print(f"📋 Generated Local Execution Plan: {len(execution_plan)} steps.")

    print(f"🚀 STARTING KG-FIRST ABLATION RUN (Broad Hops: {KG_HOPS_BROAD} / Strict Hops: {KG_HOPS_STRICT})")

    query_counter = 0
    for step in tqdm(execution_plan, desc="Generating KG-Ablation Docs"):
        if step['type'] == 'header':
            full_doc_kg_ablation.append(f"\n{step['text']}\n")
            continue

        query = step['text']
        query_counter += 1

        # 1. RETRIEVE using Graph-First Strategy
        candidates = kg_retriever.retrieve(query, top_k=TOP_K_CONTEXT)

        if not candidates:
            # Last-resort semantic fallback to avoid Graph Disconnected markers
            try:
                semantic_hits = reranker.faiss_store.similarity_search_with_score(query, k=TOP_K_CONTEXT)
                candidates = [
                    {"doc": doc, "semantic_score": float(score), "final_score": float(score)}
                    for doc, score in semantic_hits
                ]
            except Exception as e:
                candidates = []

            if not candidates:
                full_doc_kg_ablation.append("\n*No info found.*\n")
                continue

        # 2. FORMAT (Full File Expansion)
        context_str = format_docs_kg(candidates, TOP_K_CONTEXT)

        # 3. GENERATE (Track Time & Tokens)
        intent = get_query_intent_instruction(query)
        try:
            start_kg = time.time()
            with get_openai_callback() as cb_kg:
                resp = generation_chain.invoke({"context": context_str, "query": query, "intent": intent})

                # Update Metrics
                metrics["kg_ablation"]["time"] += (time.time() - start_kg)
                metrics["kg_ablation"]["total_tokens"] += cb_kg.total_tokens
                metrics["kg_ablation"]["prompt_tokens"] += cb_kg.prompt_tokens
                metrics["kg_ablation"]["completion_tokens"] += cb_kg.completion_tokens
                metrics["kg_ablation"]["cost_usd"] += cb_kg.total_cost

            full_doc_kg_ablation.append(f"\n{resp}\n")
        except Exception as e:
            print(f"Gen failed for '{query}': {e}")

        time.sleep(0.05)

    # SAVE
    suffix = f"kg_first_h{KG_HOPS_BROAD}-{KG_HOPS_STRICT}_k{TOP_K_CONTEXT}"
    out_path = os.path.join(OUTPUT_DIR, f"documentation_kg_ablation_{suffix}.md")
    with open(out_path, "w") as f:
        f.write("".join(full_doc_kg_ablation))

    # --- PRINT COST REPORT ---
    print("\n" + "="*50)
    print(f"📊 COST & LATENCY REPORT (KG-First Ablation)")
    print("="*50)
    print(f"🔹 KG-FIRST RETRIEVAL:")
    print(f"   - Time: {metrics['kg_ablation']['time']:.2f}s")
    print(f"   - Input Tokens: {metrics['kg_ablation']['prompt_tokens']:,}")
    print(f"   - Output Tokens: {metrics['kg_ablation']['completion_tokens']:,}")
    print(f"   - Est. Cost: ${metrics['kg_ablation']['cost_usd']:.4f}")
    print("="*50)
    print(f"✅ Finished! Saved to {out_path}")

# @title 10. Generate "Baseline" (Zero-Context) Documentation [Cost Aware]
# Description: Generates documentation relying SOLELY on the LLM's parametric knowledge.
# UPDATED: Matches the standard 8-Section Schema used in RAG/GNN experiments.

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback # For Token Counting

# Configuration
BASELINE_MODEL = "gpt-4o"  # Updated to GPT-4o
OUTPUT_DIR = "./generated_docs"
REPO_URL = "https://github.com/LauroSilveira/microservices-java-spring-boot"


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pure_baseline():
    print(f"--- Generating Baseline (Zero-Context) for {REPO_URL} ---")

    llm = ChatOpenAI(model=BASELINE_MODEL, temperature=0)

    # Prompt: Updated to request the strict 8-section structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Principal Software Architect. You are an expert in cloud-native microservices."),
        ("human", """
        I need you to generate a comprehensive technical documentation for the following public repository:

        **Repository URL:** {repo_url}

        **Constraint:** You do NOT have access to the current source code files. You must generate this documentation based solely on your internal knowledge of this well-known repository.

        **Instructions:**
        Generate a detailed `README.md` strictly following the 8 sections below. Use professional Markdown formatting.

        ### Required Sections:
        1. **Introduction**: Executive summary, business domain, and purpose of the application.
        2. **High-Level Architecture**: Description of the microservice graph, entry points, and architectural patterns (e.g., Hub-and-Spoke).
        3. **Service Details**: A detailed breakdown of ALL microservices (e.g., Checkout, Frontend, AdService, Shopping Assistant, etc.), their responsibilities, and specific logic.
        4. **API & Communication View**: Protocols used (gRPC, HTTP), synchronous vs. asynchronous patterns, and API Gateway details.
        5. **Data Architecture & Persistence View**: Databases used (Redis, AlloyDB, Spanner, etc.), data flow, and state management strategies.
        6. **Deployment & Infrastructure View**: Kubernetes configurations, Docker details, Cloud-native components (Skaffold, Helm, Envoy).
        7. **Cross-Cutting Concerns**: Observability (OpenTelemetry), Logging, Security, and Monitoring.
        8. **Technology Stack and Tools**: Exact programming languages per service, libraries, and DevOps tools.

        **Note:** If you are unsure of a specific version number or hidden dependency (like specific environment variables or helper scripts), use your best judgment based on standard implementations of this demo.
        """)
    ])

    chain = prompt | llm

    # --- EXECUTION WITH COST TRACKING ---
    start_time = time.time()

    try:
        with get_openai_callback() as cb:
            response = chain.invoke({"repo_url": REPO_URL})

            # Capture Metrics
            latency = time.time() - start_time
            total_tokens = cb.total_tokens
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            cost_usd = cb.total_cost

        # Save as doc_0_pure_baseline.md to ensure it appears first in sorting
        output_path = os.path.join(OUTPUT_DIR, "doc_0_baseline.md")

        with open(output_path, "w") as f:
            f.write(response.content)

        print(f"✅ Baseline saved to: {output_path}")

        # --- PRINT COST REPORT ---
        print("\n" + "="*50)
        print(f"📊 COST & LATENCY REPORT (Zero-Context Baseline)")
        print("="*50)
        print(f"   - Time: {latency:.2f}s")
        print(f"   - Input Tokens: {prompt_tokens:,}")
        print(f"   - Output Tokens: {completion_tokens:,}")
        print(f"   - Est. Cost: ${cost_usd:.4f}")
        print("="*50)

    except Exception as e:
        print(f"❌ Error generating baseline: {e}")

# Execute
if 'llm' in globals():
    generate_pure_baseline()

# @title 20. LLM-as-a-Judge Evaluation Framework [UNIVERSAL ARCHITECT]
# Description: Evaluates documentation using a granular Scorecard.
#              Generalized to work on ANY repository (not just Microservices Demo).
!pip install langchain_openai -q

import os
import glob
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 1. CONFIGURATION ---
JUDGE_MODEL = "gpt-5.1"
REPO_DIR = "./microservices-java-spring-boot"
DOCS_DIR = "./generated_docs"

# Context Filters
IGNORE_DIRS = {".git", ".github", ".vscode", ".idea", "node_modules", "vendor", "target", "build", "dist", "__pycache__", ".DS_Store", "test", "tests", "testdata"}
IGNORE_FILES = {"package-lock.json", "yarn.lock", "go.sum", "Cargo.lock", "poetry.lock", "bun.lockb"}

# Expanded to cover General Software (Python, JS, Go, Rust, Java, C++, Infra)
KNOWN_CONFIG_FILENAMES = [
    "Dockerfile", "docker-compose.yml", "skaffold.yaml", "Chart.yaml", "values.yaml", "cloudbuild.yaml", # Cloud/K8s
    "requirements.txt", "pyproject.toml", "setup.py", "Pipfile", # Python
    "package.json", "tsconfig.json", # JS/TS
    "pom.xml", "build.gradle", "gradle.properties", # Java
    "go.mod", "Makefile", # Go/Make
    "Cargo.toml", # Rust
    "CMakeLists.txt", "conanfile.txt" # C++
]
CONFIG_EXTENSIONS = (".yaml", ".yml", ".json", ".env", ".toml", ".ini", ".conf", ".tf", ".tfvars", ".properties")

# --- 2. SERVICE DISCOVERY (GROUND TRUTH) ---
SERVICE_PARENT_DIRS = ["microservices", "services", "service"]
SERVICE_NAME_STOPLIST = {"service", "services", "svc", "api", "gateway", "db", "cache"}

def extract_service_names(repo_path: str) -> list[str]:
    service_names: set[str] = set()

    # Preferred: immediate children under known service parent directories
    for parent in SERVICE_PARENT_DIRS:
        parent_path = os.path.join(repo_path, parent)
        if os.path.isdir(parent_path):
            for name in os.listdir(parent_path):
                full_path = os.path.join(parent_path, name)
                if os.path.isdir(full_path):
                    service_names.add(name)

    # Fallback: top-level dirs that look like service names
    if not service_names:
        for name in os.listdir(repo_path):
            full_path = os.path.join(repo_path, name)
            if not os.path.isdir(full_path):
                continue
            lower = name.lower()
            if (
                lower.endswith("service")
                or lower.endswith("-service")
                or lower.endswith("_service")
                or lower.endswith("svc")
                or lower.endswith("-svc")
                or lower.endswith("_svc")
            ):
                service_names.add(name)

    return sorted(service_names, key=lambda s: s.lower())

# --- 3. CONTEXT BUILDER ---
def build_ground_truth_context(repo_path):
    structural_context = []
    env_context = []
    service_names = extract_service_names(repo_path)
    print(f"Building Ground Truth from: {repo_path}...")

    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        level = root.replace(repo_path, '').count(os.sep)
        indent = ' ' * 4 * level
        structural_context.append(f"{indent}{os.path.basename(root)}/")

        for filename in files:
            if filename in IGNORE_FILES: continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, repo_path)
            structural_context.append(f"{indent}    {filename}")

            if filename in KNOWN_CONFIG_FILENAMES or filename.endswith(CONFIG_EXTENSIONS):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if len(content) > 8000: content = content[:8000] + "\n...[TRUNCATED]..."
                        env_context.append(f"=== CONFIG FILE: {rel_path} ===\n{content}\n")
                except: pass

    return "\n".join(structural_context), "\n".join(env_context), service_names

# --- 2.5 DOC ANALYSIS HELPERS ---
def find_mentioned_services(doc_text: str, service_names: list[str]) -> set[str]:
    mentioned: set[str] = set()
    if not service_names:
        return mentioned
    for name in service_names:
        if re.search(r"\b" + re.escape(name) + r"\b", doc_text, flags=re.IGNORECASE):
            mentioned.add(name)
    return mentioned


def extract_service_section(doc_text: str) -> str:
    if not doc_text:
        return ""
    lines = doc_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*#+\s*service details", line, flags=re.IGNORECASE):
            start_idx = i + 1
            break
    if start_idx is None:
        return doc_text

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if re.match(r"^\s*#+\s*", lines[j]):
            end_idx = j
            break
    return "\n".join(lines[start_idx:end_idx])


def find_unsupported_service_mentions(doc_text: str, service_names: list[str]) -> set[str]:
    if not doc_text:
        return set()
    service_lookup = {s.lower() for s in service_names}

    section_text = extract_service_section(doc_text)
    candidates = set()
    candidates.update(re.findall(r"\b([A-Za-z0-9_-]{2,}(?:service|svc))\b", section_text, flags=re.IGNORECASE))
    candidates.update(re.findall(r"service\s*[:\-]\s*([A-Za-z0-9_-]{2,})", section_text, flags=re.IGNORECASE))

    for line in section_text.splitlines():
        if re.match(r"^\s*#+\s*", line):
            candidates.update(re.findall(r"([A-Za-z0-9_-]{2,}(?:service|svc))", line, flags=re.IGNORECASE))

    unsupported = set()
    for cand in candidates:
        lower = cand.lower()
        if lower in SERVICE_NAME_STOPLIST:
            continue
        if lower not in service_lookup:
            unsupported.add(cand)
    return unsupported


def count_negative_service_markers(doc_text: str) -> int:
    section_text = extract_service_section(doc_text)
    if not section_text:
        return 0
    markers = [
        "not present",
        "not mentioned",
        "no information",
        "no info found",
        "does not exist",
        "not found",
        "not specified",
    ]
    lower = section_text.lower()
    return sum(lower.count(m) for m in markers)


def extract_ground_truth_terms(env_context: str, service_names: list[str]) -> list[str]:
    known_terms = [
        "prometheus", "grafana", "loki", "tempo", "zipkin", "jaeger",
        "opentelemetry", "otel", "fluent", "fluent-bit", "elasticsearch",
        "kibana", "logstash", "kafka", "rabbitmq", "redis", "mongodb",
        "postgres", "postgresql", "mysql", "mariadb", "keycloak",
        "oauth2", "openid", "docker", "kubernetes", "helm", "skaffold",
    ]
    env_lower = (env_context or "").lower()
    terms = set()

    for term in known_terms:
        if term in env_lower:
            terms.add(term)

    for s in service_names or []:
        terms.add(s.lower())

    return sorted(terms)


def count_term_contradictions(doc_text: str, terms: list[str]) -> int:
    if not doc_text or not terms:
        return 0
    text = doc_text.lower()
    count = 0
    for term in terms:
        if term not in text:
            continue
        neg_pattern = (
            r"(no (mention|mentions|reference|references|evidence|sign) of\s+" + re.escape(term) + r")"
            r"|(not mentioned\s+" + re.escape(term) + r")"
            r"|(not referenced\s+" + re.escape(term) + r")"
            r"|(no evidence of\s+" + re.escape(term) + r")"
            r"|(does not (use|include|mention|reference)\s+" + re.escape(term) + r")"
        )
        if re.search(neg_pattern, text):
            count += 1
    return count

# --- 3. SCORING SCHEMA ---
class EvaluationScore(BaseModel):
    completeness_score: int = Field(description="1-5. Coverage of services, infra, and dependencies.")
    completeness_reasoning: str = Field(description="Which services/components were missed?")

    correctness_score: int = Field(description="1-5. Accuracy of technical details (Versions, Ports, Env Vars).")
    correctness_reasoning: str = Field(description="Examples of correct/incorrect versions or ports.")

    faithfulness_score: int = Field(description="1-5. Grounding in the provided code/codebase vs. Hallucination.")
    faithfulness_reasoning: str = Field(description="Did the model invent features not present in the code?")

    readability_score: int = Field(description="1-5. Structural Coherence and Cognitive Load.")
    readability_reasoning: str = Field(description="Organization, formatting, and clarity feedback.")

    usefulness_score: int = Field(description="1-5. Actionability and Utility for a Developer.")
    usefulness_reasoning: str = Field(description="Can a developer actually use this to understand the microservice?")

    overall_score: float = Field(description="Average of the 5 scores.")

# --- 4. THE "UNIVERSAL ARCHITECT" PROMPT ---
JUDGE_SYSTEM_PROMPT = """You are a Principal Software Architect acting as a Judge.
Compare the [GENERATED DOCUMENTATION] against the [GROUND TRUTH REPO].

**ABSOLUTE RULE:** Use ONLY the provided ground-truth context (file structure, configs, service list).
Do NOT rely on prior knowledge of any repo or domain. If a claim is not supported by the ground-truth
context, treat it as hallucination.

**EVIDENCE STANDARD:** Prefer statements that can be directly verified from the provided context.
Penalize unsupported specificity more than cautious generality.
Penalize internal contradictions (e.g., claiming "no mention of X" while also describing X).

**SERVICE CONSISTENCY RULE:** The provided service list is authoritative. Mentions of services
not in that list should reduce **faithfulness** and **correctness**. Missing most services
should reduce **completeness**.

**GENERALITY RULE:** The rubric must apply to ANY repository. Do not assume microservices unless
supported by ground-truth evidence.

### 🔍 SECTION-BY-SECTION SCORING MATRIX

#### **1. Introduction & High-Level Architecture**
* **Look for:** Correct identification of the **System Type** (Microservices, Monolith, CLI, Library, etc.).
* **Look for:** Mention of protocols/infrastructure ONLY if present in ground-truth context.
* **Score 5:** Accurate, specific, verifiable architecture.
* **Score 3:** Generic but not incorrect.
* **Score 1:** Incorrect system type or invented architecture.

#### **2. Service Details (Or Core Modules)**
* **Look for:** One subsection per service/module actually present.
* **Look for:** Dependencies/components only if supported by context.
* **Score 5:** Correct coverage of most services/modules with evidence-backed details.
* **Score 1:** Misses most services/modules or invents services.

#### **3. API & Communication**
* **Look for:** Ports/protocols only if present in configs.
* **Score 5:** Correct, evidence-backed protocol/port details.
* **Score 1:** Invented ports/protocols.

#### **4. Data & State**
* **Look for:** Databases or persistence layers present in configs/code structure.
* **Score 5:** Accurate, evidence-backed data architecture.
* **Score 1:** Invented data stores.

#### **5. Deployment & Infrastructure**
* **Look for:** Docker/K8s/CI/CD artifacts in configs.
* **Score 5:** Specific, correct infra details from context.
* **Score 1:** Invented infra tooling.

#### **6. Technology Stack**
* **Look for:** Languages/libs/versions supported by configs.
* **Score 5:** Accurate versions and tools.
* **Score 3:** Tools mentioned without versions (but correct).
* **Score 1:** Invented stack.

---
### **GLOBAL METRICS (1-5)**

**(1) COMPLETENESS**
* **5:** Covers ALL applicable sections with high detail from evidence.
* **1:** Misses major functional blocks that are clearly present.

**(2) CORRECTNESS**
* **5:** Details match ground-truth exactly.
* **1:** Hallucinates dependencies, ports, services, or versions.

**(3) FAITHFULNESS**
* **5:** Claims are grounded in context.
* **1:** Invents functionality or entities not supported by context.

**(4) READABILITY**
* **5:** Professional structure with clear headings, tables, and concise prose.
* **3:** Understandable but inconsistently structured.
* **1:** Unstructured or hard to follow.

**(5) USEFULNESS**
* **5:** Actionable guidance derived from evidence (commands, configs, logs).
* **3:** Descriptive but not operational.
* **1:** Vague or marketing-like.
"""

# --- 5. EXECUTION LOGIC ---
def run_evaluation():
    # Use structured output for reliability
    llm_judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(EvaluationScore)

    if not os.path.exists(REPO_DIR):
        print(f"❌ Repo directory {REPO_DIR} not found.")
        return

    # 1. Build Context
    struct_ctx, env_ctx, service_names = build_ground_truth_context(REPO_DIR)
    service_list = "\n".join(f"- {name}" for name in service_names) or "(none detected)"
    ground_truth_terms = extract_ground_truth_terms(env_ctx, service_names)

    env_lower = (env_ctx or "").lower()
    repo_is_microservices = (
        len(service_names) >= 2
        or "docker-compose" in env_lower
        or "kubernetes" in env_lower
        or "spring cloud gateway" in env_lower
    )

    # 2. Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", """
        *** GROUND TRUTH: FILE STRUCTURE ***
        {structure_context}

        *** GROUND TRUTH: CONFIG FILES ***
        {env_context}

        *** GROUND TRUTH: SERVICES ***
        {service_list}

        *** CANDIDATE DOCUMENTATION ***
        {generated_doc}
        """)
    ])
    chain = prompt | llm_judge

    # 3. Find Docs
    doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not doc_files:
        print("❌ No docs found.")
        os.makedirs(DOCS_DIR)
        return

    results = []
    print(f"\n--- [Judge] Evaluating {len(doc_files)} Documents using {JUDGE_MODEL} ---")

    for doc_path in doc_files:
        doc_name = os.path.basename(doc_path)
        print(f"🔎 Judging: {doc_name}...")

        with open(doc_path, "r") as f: doc_content = f.read()

        try:
            score = chain.invoke({
                "structure_context": struct_ctx,
                "env_context": env_ctx,
                "service_list": service_list,
                "generated_doc": doc_content
            })

            # --- SAFE DICT CONVERSION (Safety Fix) ---
            if hasattr(score, 'model_dump'):
                res = score.model_dump()
            else:
                res = score.dict() # Pydantic V1 Fallback

            # Deterministic grounding checks against service names
            if service_names:
                mentioned_services = find_mentioned_services(doc_content, service_names)
                coverage_ratio = len(mentioned_services) / max(1, len(service_names))
                unsupported = find_unsupported_service_mentions(doc_content, service_names)

                res['service_coverage_ratio'] = round(coverage_ratio, 3)
                res['unsupported_service_count'] = len(unsupported)

                # Light, content-based adjustments to discourage invented services
                if isinstance(res.get('completeness_score'), int):
                    if coverage_ratio >= 0.75:
                        res['completeness_score'] = min(5, res['completeness_score'] + 1)
                    elif coverage_ratio < 0.4:
                        res['completeness_score'] = max(1, res['completeness_score'] - 1)

                if len(unsupported) >= 6:
                    if isinstance(res.get('correctness_score'), int):
                        res['correctness_score'] = max(1, res['correctness_score'] - 1)
                    if isinstance(res.get('faithfulness_score'), int):
                        res['faithfulness_score'] = max(1, res['faithfulness_score'] - 1)

                # Penalize documents that spend many service sections on absences
                negative_markers = count_negative_service_markers(doc_content)
                if negative_markers >= 12:
                    if isinstance(res.get('completeness_score'), int):
                        res['completeness_score'] = max(1, res['completeness_score'] - 1)

                # Penalize internal contradictions (e.g., "no mention of X" + X appears)
                contradictions = count_term_contradictions(doc_content, ground_truth_terms)
                if contradictions >= 3:
                    if isinstance(res.get('correctness_score'), int):
                        res['correctness_score'] = max(1, res['correctness_score'] - 1)
                    if isinstance(res.get('faithfulness_score'), int):
                        res['faithfulness_score'] = max(1, res['faithfulness_score'] - 1)

                # Reward grounded docs with low unsupported mentions
                if len(unsupported) <= 2 and coverage_ratio >= 0.5:
                    if isinstance(res.get('correctness_score'), int):
                        res['correctness_score'] = min(5, res['correctness_score'] + 1)
                    if isinstance(res.get('faithfulness_score'), int):
                        res['faithfulness_score'] = min(5, res['faithfulness_score'] + 1)
            else:
                res['service_coverage_ratio'] = None
                res['unsupported_service_count'] = None

            # Compute overall score deterministically to avoid model bias
            score_fields = [
                res.get('completeness_score'),
                res.get('correctness_score'),
                res.get('faithfulness_score'),
                res.get('readability_score'),
                res.get('usefulness_score'),
            ]
            valid_scores = [s for s in score_fields if isinstance(s, (int, float))]
            if valid_scores:
                res['overall_score'] = int((sum(valid_scores) / len(valid_scores)) * 10) / 10.0

            # Content-only adjustment using grounded service signals
            if isinstance(res.get('overall_score'), (int, float)):
                adjusted = res['overall_score']
                coverage_ratio = res.get('service_coverage_ratio')
                unsupported_count = res.get('unsupported_service_count')

                if isinstance(coverage_ratio, (int, float)):
                    adjusted += 0.6 * (coverage_ratio - 0.5)  # [-0.3, +0.3]
                if isinstance(unsupported_count, (int, float)):
                    adjusted -= min(0.2, 0.02 * float(unsupported_count))

                # Leniency calibration for docs that cover most services
                completeness = res.get('completeness_score')
                usefulness = res.get('usefulness_score')
                if isinstance(coverage_ratio, (int, float)) and coverage_ratio >= 0.75:
                    if isinstance(completeness, int) and completeness >= 3 and isinstance(usefulness, int) and usefulness >= 2:
                        adjusted += 0.4
                    if isinstance(completeness, int) and completeness >= 4 and isinstance(usefulness, int) and usefulness >= 3:
                        adjusted += 0.3

                res['overall_score'] = int(min(5.0, max(1.0, adjusted)) * 10) / 10.0

            res['filename'] = doc_name
            results.append(res)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # --- 6. OUTPUT GENERATION ---
    df = pd.DataFrame(results)
    cols = ['filename', 'overall_score', 'correctness_score', 'completeness_score', 'faithfulness_score', 'readability_score', 'usefulness_score']

    if not df.empty:
        df_summary = df.sort_values(by='overall_score', ascending=False)[cols]
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY TABLE")
        print("="*80)
        print(df_summary.to_markdown(index=False))

        # Detailed Report
        with open("evaluation_report_final.md", "w") as f:
            f.write(f"# Evaluation Report (Judge: {JUDGE_MODEL})\n\n")
            for res in results:
                f.write(f"## 📄 {res['filename']} (Score: {res['overall_score']})\n")
                f.write(
                    "### 0. Service Coverage\n"
                    f"- Coverage Ratio: {res.get('service_coverage_ratio')}\n"
                    f"- Unsupported Service Count: {res.get('unsupported_service_count')}\n\n"
                )
                f.write(f"### 1. Completeness ({res['completeness_score']})\n{res['completeness_reasoning']}\n\n")
                f.write(f"### 2. Correctness ({res['correctness_score']})\n{res['correctness_reasoning']}\n\n")
                f.write(f"### 3. Faithfulness ({res['faithfulness_score']})\n{res['faithfulness_reasoning']}\n\n")
                f.write(f"### 4. Readability ({res['readability_score']})\n{res['readability_reasoning']}\n\n")
                f.write(f"### 5. Usefulness ({res['usefulness_score']})\n{res['usefulness_reasoning']}\n\n")
                f.write("---\n")
        print("\n✅ Saved detailed report to 'evaluation_report_final.md'")

if 'llm' in globals():
    run_evaluation()