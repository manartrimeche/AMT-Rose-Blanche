-- =============================================================================
-- RAG Enzyme Search  Initialisation PostgreSQL
-- =============================================================================
-- Note: utilise double precision[] pour stocker les vecteurs.
-- La similarite cosine est calculee cote Python (numpy) pour flexibilite.

-- 2) Table principale
CREATE TABLE IF NOT EXISTS embeddings (
    id              BIGSERIAL    PRIMARY KEY,
    id_document     TEXT         NOT NULL,
    section         TEXT,
    texte_fragment  TEXT         NOT NULL,
    tokens          INT,
    vecteur         double precision[]  NOT NULL,
    tsv             tsvector
);

-- 4) Index GIN full-text sur tsvector
CREATE INDEX IF NOT EXISTS idx_embeddings_tsv
    ON embeddings
    USING gin (tsv);

-- 5) Index B-tree sur id_document
CREATE INDEX IF NOT EXISTS idx_embeddings_doc
    ON embeddings (id_document);

-- 6) Trigger : remplir tsv automatiquement a l insertion / mise a jour
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
