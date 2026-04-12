# PodSearch

Semantic podcast search using RAG and embeddings. Find episodes by content, not just title. Built with FastAPI, React, and OpenAI embeddings.

## Prerequisites

- Python 3.11+

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/PodSearch.git
   cd PodSearch
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

   Fill in your API keys in `.env`.

5. **Run the server**

   ```bash
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```

6. **Verify it's running**

   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. You should see:

   ```json
   {"message": "Hello World"}
   ```
