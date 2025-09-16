# Brain Bank CSV Chat Agent

Brain Bank is a lightweight FastAPI service that lets you upload any CSV file, map
its columns, and ask grounded questions against the rows. Under the hood the
service builds TF-IDF embeddings, indexes them with FAISS, and surfaces the most
relevant rows for every query. A minimal browser UI is bundled for quick
interactive testing.

## Features

- 📄 **CSV ingestion with flexible column mapping** – choose which fields contain
  the text to index and which ones should appear as metadata in answers.
- 🔍 **Embedding-powered retrieval** – TF-IDF embeddings are normalized and
  searched with a FAISS index for fast semantic lookups.
- 💬 **Grounded responses** – the agent never hallucinates; it simply returns the
  most relevant rows along with their metadata.
- 🖥️ **Browser playground** – upload files, run queries, and inspect matches from
  a tiny single-page UI.

## Quickstart

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the FastAPI app with Uvicorn:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Open the playground at <http://localhost:8000> to upload a CSV and start
   chatting.

## API

### `POST /load_csv`

Upload a CSV file and optionally specify comma-separated column lists.

- **Form fields**
  - `file` *(required)* – the CSV file to load.
  - `text_columns` *(optional)* – column names whose contents should be combined
    into the searchable text corpus.
  - `metadata_columns` *(optional)* – column names to include in the response
    metadata.
- **Response**

  ```json
  {
    "message": "CSV loaded successfully.",
    "rows": 128,
    "columns": ["id", "question", "answer"],
    "text_columns": ["question", "answer"],
    "metadata_columns": ["id"]
  }
  ```

### `POST /ask`

Query the current CSV.

- **Body**

  ```json
  {
    "question": "What is the capital of France?",
    "top_k": 3
  }
  ```

- **Response**

  ```json
  {
    "answer": "Top matches from the CSV:\n• Paris is the capital of France. (id: 42)",
    "matches": [
      {
        "score": 0.78,
        "row_index": 5,
        "text": "Paris is the capital of France.",
        "metadata": {"id": 42}
      }
    ]
  }
  ```

### `GET /status`

Returns information about the currently loaded dataset.

## Development notes

- The embeddings rely on scikit-learn's `TfidfVectorizer`, normalized for cosine
  similarity, and are stored in an in-memory FAISS index.
- All data is kept in-memory for simplicity; restart the server to clear the
  current session.
- The browser UI is served directly from the FastAPI app at `/` and communicates
  with the API using the same origin.

## License

MIT
