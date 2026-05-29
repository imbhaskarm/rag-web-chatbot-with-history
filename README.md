# RAG Web Chatbot with Conversation History

A conversational RAG chatbot that loads a webpage, indexes it with FAISS, and answers follow-up questions correctly by reformulating each query against the conversation history before retrieval.

Built while learning how a history-aware retriever differs from a plain retriever — and why follow-up questions like "Tell me more about it" break basic RAG without context reformulation.

---

## How It Works

```
User question
      |
      v
[contextualize_q_prompt]     -- LLM rewrites the question as a standalone query using chat history
      |
      v
[history_aware_retriever]    -- retrieves relevant chunks from FAISS
      |
      v
[question_answer_chain]      -- LLM answers using retrieved context + chat history
      |
      v
[RunnableWithMessageHistory] -- auto-stores Q/A pairs in session store
      |
      v
    Answer
```

---

## Setup

```bash
git clone https://github.com/imbhaskarm/rag-web-chatbot-with-history.git
cd rag-web-chatbot-with-history
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Add your Groq API key to `.env`. Get a free key at https://console.groq.com

> First run downloads the `sentence-transformers/all-MiniLM-L6-v2` embedding model (~90 MB). This only happens once.

---

## Run

```bash
python main.py
```

The script loads and indexes the webpage on startup (takes ~10 seconds), then starts an interactive CLI loop.

**Example session:**
```
You: What is self-reflection in autonomous agents?
Bot: Self-reflection allows agents to improve iteratively by refining past action decisions...

You: Tell me more about it
Bot: Self-reflection involves analyzing past actions to identify mistakes and improve...
# Follow-up works correctly because the question was reformulated before retrieval
```

---

## Bugs Fixed vs Original Notebook

| Bug | Original | Fix |
|---|---|---|
| Non-existent package | `from langchain_classic.chains import ...` | `from langchain.chains import ...` |
| Vector store | `Chroma` (requires separate DB process) | `FAISS` (in-memory, no setup needed) |
| Embeddings | `OllamaEmbeddings` (requires Ollama running locally) | `HuggingFaceEmbeddings` (free, local, no server) |

---

## What I Learned

- Basic RAG fails on follow-up questions like "Tell me more" because the retriever has no context. `create_history_aware_retriever` solves this by first rewriting the query into a self-contained question
- `RunnableWithMessageHistory` with `input_messages_key` and `history_messages_key` handles history injection automatically — you don't manually append to a list
- FAISS is simpler than Chroma for local projects: no server, no persistence config, just `from_documents()` and you're done

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | Chain assembly, history-aware retriever, RAG chain |
| Groq (Llama 3.3 70B) | LLM inference |
| FAISS | Vector similarity search |
| HuggingFace sentence-transformers | Free local embeddings |
| BeautifulSoup4 | HTML parsing for web loader |

---

## GitHub Topics

`langchain` `rag` `faiss` `groq` `conversational-ai`
