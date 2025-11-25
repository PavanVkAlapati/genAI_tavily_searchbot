**SOTIN: Search On Tavily If Needed**

And keep everything else professional and clean.

Here is the corrected README:

---

# SOTIN: Search On Tavily If Needed

## Overview

SOTIN is a Streamlit-based client application designed to interact with a backend news and web summarization system. The application provides a structured conversational interface, a consistent UI theme, and seamless integration with external search services. It supports chat history, citation display, and export of conversation transcripts in both Markdown and PDF formats.

The frontend communicates directly with a backend API responsible for query processing, retrieval, summarization, and citation generation.

---

## Features

### Frontend (Streamlit Client)

* Session-based conversation management
* Clean UI with custom styling
* User and assistant avatars
* Built-in suggestion prompts for quick queries
* Support for Markdown and PDF transcript export
* Automatic session identification
* Secure environment variable usage

### Backend (Expected Components)

While the backend may be kept in a separate service, the client assumes that the backend provides:

* A `/chat` POST endpoint
* Retrieval-Augmented Generation logic
* Optional external news or web search (Tavily or other services)
* Citations with title, URL, and snippet fields

---

## Project Structure

```
.
├── streamlit_client.py
├── main.py
├── agent.py
├── tools.py
├── assets/
├── data/
├── notebooks/
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
└── poetry.lock
```

---

## Requirements

### Python Version

Python 3.9 or later is recommended.

### Install with Poetry

```
poetry install
```

### Install with pip

```
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```
NEWS_AGENT_API_URL=http://127.0.0.1:8000/chat
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

The `.env` file contains sensitive information and must not be committed to GitHub.

---

## Running the Application

### Start Backend

Start your backend server that exposes the `/chat` endpoint:

```
python main.py
```

### Start Streamlit Interface

```
streamlit run streamlit_client.py
```

---

## API Contract

### Request Payload

```json
{
  "user_id": "string",
  "session_id": "string",
  "message": "user text"
}
```

### Expected Response

```json
{
  "final_answer": "assistant response text",
  "citations": [
    {
      "title": "optional",
      "url": "optional",
      "snippet": "optional"
    }
  ]
}
```

---

## Transcript Export

The application supports:

* Export to Markdown (.md)
* Export to PDF (.pdf) using ReportLab (plain text rendering)

If ReportLab is not installed, PDF export is disabled automatically.

---

## Assets

The following icons should be present in the `assets/` directory:

* `logo.png`
* `user.png`
* `bot.png`
* `send.png`
* `download.png`

These are used for UI consistency and branding.

---

## License

standard MIT licence
