-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the MemoryItems table to store all information nodes
CREATE TABLE MemoryItems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id UUID REFERENCES MemoryItems(id) ON DELETE SET NULL,
    content_type VARCHAR(50) NOT NULL,
    text_content TEXT,
    analyzed_text TEXT,
    data_uri TEXT,
    embedding HALFVEC(2560),
    embedding_model_version VARCHAR(100),
    meta JSONB,
    event_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create the Relationships table to connect the nodes
CREATE TABLE Relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id UUID NOT NULL REFERENCES MemoryItems(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES MemoryItems(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add a trigger to the MemoryItems table to call the function on update
CREATE TRIGGER set_timestamp
BEFORE UPDATE ON MemoryItems
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();


-- Add indexes for performance
-- IVFFlat index with halfvec for efficient half-precision vector similarity search
CREATE INDEX ON MemoryItems USING hnsw (embedding halfvec_l2_ops) WITH (m = 16, ef_construction = 64);

-- Index on content_type for faster filtering
CREATE INDEX ON MemoryItems (content_type);

-- Index on event_timestamp for time-based queries
CREATE INDEX ON MemoryItems (event_timestamp);

-- Indexes on foreign keys in the Relationships table
CREATE INDEX ON Relationships (source_node_id);
CREATE INDEX ON Relationships (target_node_id);

-- A composite index can be useful for querying relationships by type
CREATE INDEX ON Relationships (source_node_id, relationship_type);
