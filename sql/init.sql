-- =============================================================================
-- RAG Enzyme Search — Initialisation PostgreSQL + pgvector
-- =============================================================================

-- 1) Extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Table principale
CREATE TABLE IF NOT EXISTS embeddings (
    id              BIGSERIAL    PRIMARY KEY,
    id_document     TEXT         NOT NULL,
    section         TEXT,
    texte_fragment  TEXT         NOT NULL,
    tokens          INT,
    vecteur         VECTOR(384)  NOT NULL,
    tsv             tsvector
);

-- 3) Index vectoriel HNSW (cosine) — optimal pour top-k rapide
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings
    USING hnsw (vecteur vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- 4) Index GIN full-text sur tsvector
CREATE INDEX IF NOT EXISTS idx_embeddings_tsv
    ON embeddings
    USING gin (tsv);

-- 5) Index B-tree sur id_document
CREATE INDEX IF NOT EXISTS idx_embeddings_doc
    ON embeddings (id_document);

-- 6) Trigger : remplir tsv automatiquement à l'insertion / mise à jour
CREATE OR REPLACE FUNCTION embeddings_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('simple', COALESCE(NEW.texte_fragment, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_embeddings_tsv ON embeddings;

CREATE TRIGGER trg_embeddings_tsv
    BEFORE INSERT OR UPDATE ON embeddings
    FOR EACH ROW EXECUTE FUNCTION embeddings_tsv_trigger();
