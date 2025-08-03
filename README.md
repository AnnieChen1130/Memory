# Memory System API

A backend service for ingesting, storing, and retrieving event logs with semantic search capabilities.

## Features

- Ingest various content types (text, images, audio, video, web links)
- Semantic search with embedding-based retrieval
- Background processing for complex content
- Relationship tracking between memory items
- Reranking for improved search results

## Getting Started

We recommend using [nix](https://nixos.org/download/#nix-install-linux) (with `Flakes` enabled) and [uv](https://docs.astral.sh/uv/) for running this project. [direnv](https://direnv.net/) is also preferred.

clone this repo first: `git clone https://github.com/Sunny-XXV/Memroy`

```shell
cd memory
nix develop
```

### Set up PostgreSQL for vector database

we've provided a command for initializing `PostgreSQL` with [pgvector](https://github.com/pgvector/pgvector) in `flake.nix`, just run:
```shell
start-db
```
Which would:
1. Initialize a project-specific PostgreSQL server under the root dir of this project, default to `./pg_data`, shipped with pgvector;
2. Start the Database server, at port `5432` by default.

if successful, you'll see hint on how to test the database in your terminal. After that, you can run:
```shell
psql -h ./pg_data -U ${USER} -d ${USER} -f sql/init_tables.sql
```
to create tables this project relies on.

What you can customize:
1. change the port on which the database service listen;

Change it in `./.envrc` if you're using direnv or export it first in you shell. the environment variable is `PGPORT`.

### Set up MinIO for object storage

A command is provided in `flake.nix` for MinIO initialization as well, run:
```shell
start-minio
```
Which would:
1. Initialize a MinIO instance under the root dir of this project, default to `./minio_data`

note that this command would take over the terminal, so you might want to run it in tmux or in the background.

What you can customize:
1. The port on which Minio listens: `MINIO_PORT`
2. Port for access to MinIO dashboard: `MINIO_CONSOLE_PORT`
3. Your user name for access to MinIO: `MINIO_ACCESS_KEY`
4. Your password for access to MinIO: `MINIO_SECRET_KEY`

either by changing them in `./.envrc` or export in shell.

### Start the Memory service

Just copy and customize **your own** `./.env` file and then run:
```
uv sync
uv run main.py
```
and the service is ready.

What you can customize (in `./.env`, for which you can get a template by `cp .env.example .env`):
1. Your database url for access, which relies on your former steps on Setting up the PostgreSQL database.
2. MinIO-related information, starting with prefix `MEMORY_MINIO_`. keep your access key and secret key safe 😉
3. Host and port for accessing the **Memroy Service**, with prefix `MEMORY_API_`
4. Model for embedding and reranking. don't change the embedding model/dimension unless you are ready for **re-calculating the embedding vectors for all your memory items!!!**
5. other things. see in `.env.example`


## API Endpoints

### Health Check

**GET** `/health`

Returns the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Ingestion

**POST** `/api/v1/ingest`

Ingest a new memory item into the system. Generates embeddings and stores the item. For complex content types, triggers background processing.

**Request Body:** `MemoryItemRaw`
```json
{
  "content_type": "text",
  "text_content": "This is a sample text content",
  "data_uri": null,
  "event_timestamp": "2024-01-15T10:30:00Z",
  "meta": {"source": "api", "user_id": "123"},
  "reply_to_id": null
}
```

**Content Types:**
- `text` - Plain text content
- `image` - Image content (requires `data_uri`)
- `audio` - Audio content (requires `data_uri`)
- `video` - Video content (requires `data_uri`)
- `web_link` - Web page content (requires `data_uri`)

**Response:** `IngestionResponse`
```json
{
  "status": "ingested",
  "item_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Examples:**

1. **Text Content:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "text",
    "text_content": "Meeting notes: Discussed project timeline and deliverables",
    "event_timestamp": "2024-01-15T14:30:00Z",
    "meta": {"meeting_id": "MTG-001", "participants": ["Alice", "Bob"]}
  }'
```

2. **Web Link:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "web_link",
    "data_uri": "https://example.com/article",
    "event_timestamp": "2024-01-15T14:30:00Z",
    "meta": {"source": "bookmark"}
  }'
```

### Retrieval

**GET** `/api/v1/retrieve`

Retrieve memory items using semantic search.

**Query Parameters:**
- `query` (required): Search query string
- `top_k` (optional, default=10): Number of results to return
- `filters` (optional): JSON string for metadata filtering
- `start_date` (optional): Filter by event timestamp start
- `end_date` (optional): Filter by event timestamp end
- `content_types` (optional): Comma-separated list of content types
- `include_context` (optional, default=false): Include related items
- `enable_reranking` (optional, default=true): Enable result reranking

**Response:** `RetrievalResponse`
```json
{
  "query": "meeting notes",
  "results": [
    {
      "item": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "parent_id": null,
        "content_type": "text",
        "text_content": "Meeting notes: Discussed project timeline",
        "analyzed_text": "Meeting notes: Discussed project timeline",
        "data_uri": null,
        "embedding": [0.1, 0.2, ...],
        "embedding_model_version": "sentence-transformers/all-MiniLM-L6-v2",
        "meta": {"meeting_id": "MTG-001"},
        "event_timestamp": "2024-01-15T14:30:00Z",
        "created_at": "2024-01-15T14:31:00Z",
        "updated_at": "2024-01-15T14:31:00Z"
      },
      "score": 0.85
    }
  ]
}
```

**Examples:**

1. **Basic Search:**
```bash
curl "http://localhost:8000/api/v1/retrieve?query=project%20timeline&top_k=5"
```

2. **Filtered Search:**
```bash
curl "http://localhost:8000/api/v1/retrieve?query=meeting&content_types=text,long_text&start_date=2024-01-01T00:00:00Z&filters=%7B%22meeting_id%22:%22MTG-001%22%7D"
```

3. **Search with Context:**
```bash
curl "http://localhost:8000/api/v1/retrieve?query=project&include_context=true"
```

### Memory Items

**GET** `/api/v1/items/{item_id}`

Retrieve a specific memory item by its ID.

**Response:** `MemoryItem`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "parent_id": null,
  "content_type": "text",
  "text_content": "Meeting notes: Discussed project timeline",
  "analyzed_text": "Meeting notes: Discussed project timeline",
  "data_uri": null,
  "embedding": [0.1, 0.2, ...],
  "embedding_model_version": "sentence-transformers/all-MiniLM-L6-v2",
  "meta": {"meeting_id": "MTG-001"},
  "event_timestamp": "2024-01-15T14:30:00Z",
  "created_at": "2024-01-15T14:31:00Z",
  "updated_at": "2024-01-15T14:31:00Z"
}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/items/550e8400-e29b-41d4-a716-446655440000"
```

**GET** `/api/v1/items/{item_id}/related`

Get items related to a specific memory item.

**Query Parameters:**
- `relationship_types` (optional): Comma-separated list of relationship types to filter

**Response:**
```json
{
  "item_id": "550e8400-e29b-41d4-a716-446655440000",
  "related_items": [
    {
      "item": {
        "id": "660e8400-e29b-41d4-a716-446655440001",
        "content_type": "text",
        "text_content": "Follow-up action items from the meeting"
      },
      "relationship": {
        "id": "770e8400-e29b-41d4-a716-446655440002",
        "source_node_id": "550e8400-e29b-41d4-a716-446655440000",
        "target_node_id": "660e8400-e29b-41d4-a716-446655440001",
        "relationship_type": "follow_up",
        "created_at": "2024-01-15T14:35:00Z"
      }
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/items/550e8400-e29b-41d4-a716-446655440000/related?relationship_types=follow_up,reply_to"
```

### Task Status

**GET** `/api/v1/tasks/{task_id}`

Get the status of a background processing task.

**Response:**
```json
{
  "task_id": "880e8400-e29b-41d4-a716-446655440003",
  "task_type": "web_scraping",
  "status": "completed",
  "source_item_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T14:31:00Z",
  "started_at": "2024-01-15T14:31:05Z",
  "completed_at": "2024-01-15T14:31:30Z",
  "error_message": null
}
```

**Task Types:**
- `web_scraping` - Processing web links
- `media_analysis` - Processing images, audio, video
- `text_analysis` - Processing long text content

**Task Statuses:**
- `pending` - Task queued but not started
- `running` - Task currently being processed
- `completed` - Task finished successfully
- `failed` - Task encountered an error

**Example:**
```bash
curl "http://localhost:8000/api/v1/tasks/880e8400-e29b-41d4-a716-446655440003"
```

## Data Models

### MemoryItemRaw (Input)
```python
{
  "content_type": str,              # Required: content type
  "text_content": str | None,       # Text content
  "data_uri": str | None,           # URI for binary data
  "event_timestamp": datetime,      # Required: when event occurred
  "meta": dict | None,              # Metadata dictionary
  "reply_to_id": UUID | None        # ID of item this replies to
}
```

### MemoryItem (Storage/Output)
```python
{
  "id": UUID,                       # Auto-generated unique ID
  "parent_id": UUID | None,         # Parent item ID
  "content_type": str,              # Content type
  "text_content": str | None,       # Original text content
  "analyzed_text": str | None,      # Processed text content
  "data_uri": str | None,           # URI for binary data
  "embedding": List[float],         # Vector embedding
  "embedding_model_version": str,   # Model version used
  "meta": dict | None,              # Metadata dictionary
  "event_timestamp": datetime,      # When event occurred
  "created_at": datetime,           # When item was created
  "updated_at": datetime            # When item was last updated
}
```

### Relationship
```python
{
  "id": UUID,                       # Unique relationship ID
  "source_node_id": UUID,           # Source item ID
  "target_node_id": UUID,           # Target item ID
  "relationship_type": str,         # Type of relationship
  "created_at": datetime            # When relationship was created
}
```

## Usage Patterns

### 1. Basic Text Ingestion and Search
```bash
# Ingest text
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "text",
    "text_content": "Important project meeting discussion",
    "event_timestamp": "2024-01-15T10:00:00Z"
  }'

# Search for it
curl "http://localhost:8000/api/v1/retrieve?query=project%20meeting"
```

### 2. Web Content Processing
```bash
# Ingest web link (triggers background processing)
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "web_link",
    "data_uri": "https://news.example.com/article",
    "event_timestamp": "2024-01-15T10:00:00Z"
  }'

# Check processing status
curl "http://localhost:8000/api/v1/tasks/{task_id}"
```

### 3. Contextual Search
```bash
# Search with related items included
curl "http://localhost:8000/api/v1/retrieve?query=project&include_context=true"
```

## Error Responses

All endpoints return standard HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable

Error response format:
```json
{
  "detail": "Error description"
}
```
