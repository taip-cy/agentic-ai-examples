from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel, GradioUI, LiteLLMModel, TransformersModel, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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

reasoner_agent = CodeAgent(
    tools=[],
    model=get_model(),
    add_base_tools=False,
    max_steps=2
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: This user's question to query the vector database with.
    """
    # Search relevant documents
    docs = vectordb.similarity_search(user_query, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    #Create prompt with context
    prompt = f""" Based on the follwoing context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
    Context: {context}

    Question: {user_query}

    Answer: 
    
    """

    response = reasoner_agent.run(prompt, reset=False)
    return response

tool_model = get_model()
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model = tool_model, add_base_tools = False, max_steps=3)

GradioUI(primary_agent).launch()