# The Library

The Library is an LLM chat using embedded data for Kill Z-R0. Inspired by KNZO's project of the same name, it aspires to demonstrate the capabilities of LLMs + vectorstores for RAG.

## Prereqs

- Install [Python](https://www.python.org/downloads/)
- Install [Ollama](https://ollama.com) and create an account
- Create a [Pinecone](https://www.pinecone.io) account

## Getting Started

- If running on MacOS, install `libmagic`:

  ```bash
  brew install libmagic
  ```

- After pulling the code, navigate to the directory and install the Python dependencies with:

  ```bash
  pip install -r requirements.txt
  ```

- Once installed, create a file named `.env` at the root of the project

- Inside `.env`, set your Pinecone API key:

  ```bash
  PINECONE_API_KEY="YOUR API KEY HERE"
  ```

  - You can define any other environment variables here as needed (Ex: If you decide to use OpenAI instead)

- Make a folder called `contexts` at the root of the project. This is where all of your context data will go.

- Change the `DirectoryLoader` glob as needed. Right now it will search recursively through the `contexts` folder and load any `.txt` file. If you decide to use a different file type like Markdown or PDF, consider updating the code to use the appropriate loader. More information about available loaders can be found in the [Langchain documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/).

- Create or move your context files into the `contexts` folder. This can include any subdirectories for organizational purposes.

- Execute `main.py`. This will load your data, embed it, and start a chat with the LLM:

  ```bash
  python main.py
  ```
