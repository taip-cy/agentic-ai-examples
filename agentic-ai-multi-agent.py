from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel, GradioUI, LiteLLMModel, TransformersModel
from custom_tools import NVDCveDetailsLookupTool
from dotenv import load_dotenv
import litellm
import os

litellm._turn_on_debug()

load_dotenv(override=True)

def get_model():
    provider = os.getenv("INFERENCE_PROVIDER").lower()

    if provider == "huggingface":
        use_local = os.getenv("HUGGINGFACE_LOCAL").lower() == "true"
        model_id = os.getenv("HUGGINGFACE_MODEL_ID")
        if use_local:
            return TransformersModel(model_id=model_id)
        else:
            api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            return HfApiModel(model_id=model_id, token=api_token)
        
    elif provider == "bedrock":
        bedrock_model_id = os.getenv("BEDROCK_MODEL_ID")
        return LiteLLMModel(
            model_id=bedrock_model_id,
            drop_params=True
        )
    
    elif provider == "ollama":
        ollama_model_id = os.getenv("OLLAMA_MODEL_ID")
        ollama_api_base = os.getenv("OLLAMA_API_BASE")
        ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX"))
        ollama_max_new_tokens = int(os.getenv("OLLAMA_MAX_NEW_TOKENS"))
        return LiteLLMModel(
            model_id=ollama_model_id,
            api_base=ollama_api_base,
            num_ctx=ollama_num_ctx,
            max_new_tokens=ollama_max_new_tokens
        )
    
    else:
        raise ValueError(f"Unsupported INFERENCE_PROVIDER: {provider}")

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=get_model(),
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

nvd_api_agent = ToolCallingAgent(
    tools=[NVDCveDetailsLookupTool()],
    model=get_model(),
    name="nvd_api_agent",
    description="Calls the NVD API and returns the CVE details from the NVD API in JSON format for you. Give it the CVE ID as an argument.",
)

manager_agent = CodeAgent(
    tools=[],
    model=get_model(),
    managed_agents=[web_agent, nvd_api_agent]
)

# will write python script and execute it to achieve the goal
# response = manager_agent.run("What is the 10th number in the Fibonacci sequence ?")

# will use the nvd_api_agent and the web_agent to achieve the goal
# response = manager_agent.run("What is the CVSS score of CVE-2025-24085 ?")

# will create a local chat interface for us to interact with the agent, we can access it at http://127.0.0.1:7860/
GradioUI(manager_agent).launch()
