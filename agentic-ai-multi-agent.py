from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel, GradioUI
from custom_tools import NVDCveDetailsLookupTool

# will use the default model from the huggingface api
model = HfApiModel()

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

nvd_api_agent = ToolCallingAgent(
    tools=[NVDCveDetailsLookupTool()],
    model=model,
    name="nvd_api_agent",
    description="Calls the NVD API and returns the CVE details from the NVD API in JSON format for you. Give it the CVE ID as an argument.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent, nvd_api_agent]
)

# will write python script and execute it to achieve the goal
# response = manager_agent.run("What is the 10th number in the Fibonacci sequence ?")

# will use the nvd_api_agent and the web_agent to achieve the goal
# response = manager_agent.run("What is the CVSS score of CVE-2025-24085 ?")

# will create a local chat interface for us to interact with the agent, we can access it at http://127.0.0.1:7860/
GradioUI(manager_agent).launch()
