#!/usr/bin/env python3

import os
import argparse
import asyncio
import nest_asyncio
from typing import Optional, Dict, Any
from IPython.display import display, HTML
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Apply nested asyncio to allow nested event loops
nest_asyncio.apply()

# Import LlamaIndex components
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)

# Import LlamaParse for document parsing
try:
    from llama_parse import LlamaParse
except ImportError:
    print("LlamaParse not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "llama-parse"])
    from llama_parse import LlamaParse

class QueryEvent(Event):
    query: str

class RAGWorkflow(Workflow):
    def __init__(self, timeout: int = 120, verbose: bool = True, storage_dir: str = "./storage", force_reindex: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.storage_dir = storage_dir
        self.force_reindex = force_reindex
        self.llm = None
        self.query_engine = None

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        if not ev.resume_file:
            raise ValueError("No resume file provided")

        # Define an LLM to work with
        self.llm = OpenAI(model="gpt-4o-mini", api_key=ev.openai_api_key)

        # Ingest the data and set up the query engine
        if os.path.exists(self.storage_dir) and not self.force_reindex:
            # You've already ingested your documents
            print("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            # Parse and load your documents
            print(f"Parsing document: {ev.resume_file}")
            documents = LlamaParse(
                api_key=ev.llama_cloud_api_key,
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
            ).load_data(ev.resume_file)
            
            print("Creating vector index...")
            # Embed and index the documents
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=OpenAIEmbedding(
                    model_name="text-embedding-3-small",
                    api_key=ev.openai_api_key
                )
            )
            
            print(f"Saving index to {self.storage_dir}")
            os.makedirs(self.storage_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=self.storage_dir)

        # Either way, create a query engine
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        # Now fire off a query event to trigger the next step
        return QueryEvent(query=ev.query)

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        print(f"Querying: {ev.query}")
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        return StopEvent(result=response.response)

def create_query_engine(resume_file: str, openai_api_key: str, llama_cloud_api_key: str, storage_dir: str = "./storage", force_reindex: bool = False):
    """Create a query engine from a resume file or load from storage if available."""
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    
    if os.path.exists(storage_dir) and not force_reindex:
        # Load the index from disk
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
    else:
        # Parse and load documents
        print(f"Parsing document: {resume_file}")
        documents = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown",
            content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
        ).load_data(resume_file)
        
        print("Creating vector index...")
        # Create a vector store index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OpenAIEmbedding(
                model_name="text-embedding-3-small",
                api_key=openai_api_key
            )
        )
        
        print(f"Saving index to {storage_dir}")
        os.makedirs(storage_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=storage_dir)
    
    # Create a query engine
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    return query_engine

def create_agent(query_engine, openai_api_key: str):
    """Create an agent that can answer questions about a resume."""
    # Create a function to answer questions about the resume
    def query_resume(q: str) -> str:
        """Answers questions about a specific resume."""
        response = query_engine.query(f"This is a question about the specific resume we have in our database: {q}")
        return response.response
    
    # Create a tool from the function
    resume_tool = FunctionTool.from_defaults(fn=query_resume)
    
    # Create an agent with the tool
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    agent = FunctionCallingAgent.from_tools(
        tools=[resume_tool],
        llm=llm,
        verbose=True
    )
    
    return agent

async def run_workflow(resume_file: str, query: str, openai_api_key: str, llama_cloud_api_key: str, 
                      storage_dir: str = "./storage", force_reindex: bool = False):
    """Run the RAG workflow with the given resume file and query."""
    w = RAGWorkflow(timeout=120, verbose=True, storage_dir=storage_dir, force_reindex=force_reindex)
    result = await w.run(
        resume_file=resume_file,
        query=query,
        openai_api_key=openai_api_key,
        llama_cloud_api_key=llama_cloud_api_key
    )
    return result

def get_interactive_query() -> str:
    """Prompt the user for a query interactively."""
    print("\nEnter your question about the resume (or 'quit' to exit):")
    query = input("> ").strip()
    if query.lower() in ['quit', 'exit']:
        raise SystemExit(0)
    return query

def main():
    parser = argparse.ArgumentParser(description="Run a RAG workflow on a resume file.")
    parser.add_argument("--resume-file", type=str, default="resume.pdf", 
                       help="Path to the resume file (default: resume.pdf)")
    parser.add_argument("--query", type=str, required=False,
                       help="Query to ask about the resume (if not provided, will prompt interactively)")
    parser.add_argument("--openai-api-key", type=str, 
                       default=os.getenv('OPENAI_API_KEY'),
                       help="OpenAI API key (can be set via OPENAI_API_KEY env variable)")
    parser.add_argument("--llama-cloud-api-key", type=str,
                       default=os.getenv('LLAMA_CLOUD_API_KEY'),
                       help="LlamaCloud API key (can be set via LLAMA_CLOUD_API_KEY env variable)")
    parser.add_argument("--storage-dir", type=str, default="./storage", help="Directory to store the index")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing even if index exists")
    parser.add_argument("--method", choices=["workflow", "agent", "query"], default="workflow", 
                        help="Method to use: workflow, agent, or direct query")
    
    args = parser.parse_args()
    
    # Get query interactively if not provided
    if not args.query:
        args.query = get_interactive_query()
    
    # Validate required API keys
    if not args.openai_api_key:
        raise ValueError("OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable")
    if not args.llama_cloud_api_key:
        raise ValueError("LlamaCloud API key must be provided via --llama-cloud-api-key or LLAMA_CLOUD_API_KEY environment variable")

    if args.method == "workflow":
        print("Running workflow...")
        result = asyncio.run(run_workflow(
            resume_file=args.resume_file,
            query=args.query,
            openai_api_key=args.openai_api_key,
            llama_cloud_api_key=args.llama_cloud_api_key,
            storage_dir=args.storage_dir,
            force_reindex=args.force_reindex
        ))
        print("\nWorkflow Result:")
        print(result)
    
    elif args.method == "agent":
        print("Creating agent...")
        query_engine = create_query_engine(
            resume_file=args.resume_file,
            openai_api_key=args.openai_api_key,
            llama_cloud_api_key=args.llama_cloud_api_key,
            storage_dir=args.storage_dir,
            force_reindex=args.force_reindex
        )
        agent = create_agent(query_engine, args.openai_api_key)
        
        print("Asking agent...")
        response = agent.chat(args.query)
        print("\nAgent Response:")
        print(response)
    
    else:  # direct query
        print("Creating query engine...")
        query_engine = create_query_engine(
            resume_file=args.resume_file,
            openai_api_key=args.openai_api_key,
            llama_cloud_api_key=args.llama_cloud_api_key,
            storage_dir=args.storage_dir,
            force_reindex=args.force_reindex
        )
        
        print("Querying...")
        response = query_engine.query(f"This is a question about the specific resume we have in our database: {args.query}")
        print("\nQuery Result:")
        print(response)

if __name__ == "__main__":
    main()
