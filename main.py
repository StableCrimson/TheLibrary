import os
import time
from models import verify_models_pulled, CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME
from argparse import ArgumentParser
from rich import print
from dotenv import load_dotenv
from typing_extensions import List, TypedDict
from pinecone import Pinecone, ServerlessSpec
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_pinecone.vectorstores import Pinecone as PineconeVectorStore

class State(TypedDict):
  question: str
  context: List[Document]
  answer: str

INDEX_NAME = "the-library"

# Load the Pinecone API key
load_dotenv()

# Make sure we have the Ollama models before we continue
verify_models_pulled()

# Load the contexts of the context folder
loader = DirectoryLoader(path="./contexts", glob="**/*.txt", show_progress=True, loader_cls=TextLoader)

print("Loading context documents...")
docs = loader.load()
print(f"[green]Loaded {len(docs)} documents!")

# Split the text content so that it fits withing the context window
# TODO: Optimize chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

print("Splitting document contents to fit within the context window...")
splits = text_splitter.split_documents(docs)
print(f"[green]Split into {len(splits)} chunks!")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

indexes = pc.list_indexes()

# If index exists, delete it
if any(index["name"] == INDEX_NAME for index in indexes.indexes):
  print("Index exists, deleting...")
  pc.delete_index(INDEX_NAME)
  print("[green]Index deleted!")

print("Creating new index...")
pc.create_index(
  name=INDEX_NAME,
  dimension=1024,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  ),
  deletion_protection="disabled"
)
print("[green]Created! Waiting for index to become ready for use...")

while not pc.describe_index(INDEX_NAME).index.status.ready:
  time.sleep(0.5)

print("[green]Index ready! Creating embedding...")


# This is an LLM that I have installed and running locally on my Mac
# So the response speed is not as fast as services like ChatGPT or Gemini
llm = ChatOllama(model=CHAT_MODEL_NAME)
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
vector_store = PineconeVectorStore(index=pc.Index(INDEX_NAME), embedding=embedding)

print("Loading context data into Pinecone vector store...")
vector_store.add_documents(documents=splits)
print("[green]Embedded!")

# Grab an established RAG prompt instead of writing our own
prompt = hub.pull("rlm/rag-prompt")

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

while True:

  print("\n")
  question = input("Query > ")
  print("\n")

  for message, _ in graph.stream({"question": question}, stream_mode="messages"):
    print(f"[blue]{message.content}", end='')
