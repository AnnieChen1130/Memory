-- Drop indexes first, as some depend on tables
-- Note: Dropping a table automatically drops its indexes, but explicit dropping is clearer
DROP INDEX IF EXISTS memoryitems_embedding_hnsw_idx;
DROP INDEX IF EXISTS memoryitems_content_type_idx;
DROP INDEX IF EXISTS memoryitems_event_timestamp_idx;
DROP INDEX IF EXISTS relationships_source_node_id_idx;
DROP INDEX IF EXISTS relationships_target_node_id_idx;
DROP INDEX IF EXISTS relationships_source_node_id_relationship_type_idx;

-- Drop the trigger from the MemoryItems table
DROP TRIGGER IF EXISTS set_timestamp ON MemoryItems;

-- Drop the function that the trigger uses
DROP FUNCTION IF EXISTS trigger_set_timestamp();

-- Drop the tables. The order matters if there are foreign key dependencies.
-- Drop Relationships first because it has a foreign key reference to MemoryItems
DROP TABLE IF EXISTS Relationships;
DROP TABLE IF EXISTS MemoryItems;