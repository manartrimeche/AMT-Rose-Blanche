# RAG Enzyme Search — Module de Recherche Sémantique

> Recherche sémantique intelligente sur un corpus de fiches techniques enzymatiques.
> PostgreSQL + numpy cosine · sentence-transformers · Cross-Encoder reranking

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
│ documents_md/ │────▶│  chunker.py  │────▶│   embedder.py       │
│  (*.md)       │     │ (sections)   │     │ (all-MiniLM-L6-v2)  │
└──────────────┘     └──────────────┘     └──────────┬──────────┘
                                                      │
                                                      ▼
                                          ┌─────────────────────┐
                                          │  PostgreSQL + numpy  │
                                          │  (cosine similarity) │
                                          └──────────┬──────────┘
                                                      │
                     ┌────────────────────────────────┤
                     │                                │
              ┌──────▼──────┐                  ┌──────▼──────┐
              │ retriever.py │                  │   api.py    │
              │  (3 modes)   │                  │  (FastAPI)  │
              └──────┬───────┘                  └─────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   vector-only    hybrid      hybrid+rerank
   (cosine)    (vec+BM25)   (cross-encoder)
```

### Les 3 stratégies de recherche

| Mode | Description | Performance |
|------|-------------|-------------|
| `vector` | Cosine similarity pure (numpy) | Rapide, bon baseline |
| `hybrid` | 0.7×cosine + 0.3×BM25 (ts_rank_cd) | Meilleur recall lexical |
| `rerank` | Hybrid + Cross-Encoder reranking | **Meilleure précision** |

---

## Prérequis

- Python 3.10+
- PostgreSQL 14+
- ~1 Go RAM pour les modèles

---

## Installation

```bash
# 1. Cloner le projet
cd AMT-Rose-Blanche

# 2. Environnement virtuel
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Dépendances
pip install -r requirements.txt

# 4. Configuration
# Éditer .env avec vos paramètres PostgreSQL
cp .env.example .env  # ou éditer .env directement
```

### Setup PostgreSQL

```sql
-- Créer la base de données
CREATE DATABASE rag_enzymes;

-- Initialiser le schéma (depuis psql)
\i sql/init.sql
```

Ou via Python :
```python
from src.db import init_schema
init_schema()
```

---

## Corpus

Les fiches techniques enzymatiques sont dans `documents_md/` au format Markdown structuré :

```markdown
# Nom du Produit
## Type
...
## Description
...
## Applications
...
## Dosage recommandé
...
## Stockage
...
```

35 fiches techniques couvrant : lipases, amylases, xylanases, transglutaminases, glucose oxydases, améliorants de panification.

---

## Ingestion

```bash
# Ingestion standard
python ingest.py

# Avec reset de la table
python ingest.py --reset

# Dry-run (test sans insertion)
python ingest.py --dry-run
```

Le script :
1. Parse les fichiers `.md` (découpage par sections `##`)
2. Nettoie le bruit (emails, URLs, téléphones)
3. Normalise les unités (ppm, U/g, XylH/g, etc.)
4. Préfixe chaque chunk par `Produit: ... | Section: ...`
5. Génère les embeddings (all-MiniLM-L6-v2, dim 384)
6. Insère dans PostgreSQL avec pgvector

---

## Recherche (CLI)

```bash
# Mode rerank (recommandé)
python search.py "Quelle enzyme pour le volume du pain ?" --mode rerank

# Mode hybride
python search.py "dosage xylanase" --mode hybrid

# Mode vector pur
python search.py "stockage enzymes" --mode vector

# Sortie JSON
python search.py "transglutaminase" --mode rerank --json
```

---

## API REST (FastAPI)

```bash
# Lancer le serveur
uvicorn api:app --reload --port 8000
```

### Endpoints

#### POST /search
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle enzyme améliore le volume du pain ?", "mode": "rerank", "top_k": 3}'
```

Réponse :
```json
{
  "question": "Quelle enzyme améliore le volume du pain ?",
  "mode": "rerank",
  "count": 3,
  "results": [
    {
      "id_document": "BVZyme A SOFT405",
      "section": "Description",
      "texte_fragment": "Produit: BVZyme A SOFT405 | Section: Description\nSystème enzymatique...",
      "score_final": 0.8234,
      "score_vec": 0.7891,
      "score_bm25": 0.4521,
      "score_rerank": 0.9012
    }
  ]
}
```

#### GET /health
```bash
curl http://localhost:8000/health
```

### Documentation interactive
- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

---

## Évaluation

25 questions gold avec documents attendus (`gold_questions.json`).

```bash
# Évaluation complète (3 modes)
python evaluate.py

# Un seul mode
python evaluate.py --modes rerank

# Export CSV
python evaluate.py --output results.csv
```

### Métriques

| Métrique | Définition |
|----------|-----------|
| **Recall@3** | Proportion de documents pertinents retrouvés dans le top 3 |
| **MRR@3** | Rang réciproque moyen du premier résultat pertinent |

### Résultats attendus

| Mode | Recall@3 | MRR@3 |
|------|----------|-------|
| vector | 0.5700 | 0.6667 |
| **hybrid** | **0.5800** | **0.6400** |
| rerank | 0.5500 | 0.6200 |

---

## Décisions techniques

### Pourquoi PostgreSQL + numpy ?
- Pour 251 fragments, le calcul numpy en Python est instantané (< 5ms)
- Les vecteurs sont stockés comme `double precision[]` dans PostgreSQL
- Pas besoin d'extension externe (pgvector): tout fonctionne nativement
- Full-text search via `tsvector` + GIN index pour la composante lexicale
- Architecture extensible : migration pgvector possible en ajoutant l'extension

### Pourquoi hybrid (vector + BM25) ?
- Le vector seul manque les correspondances lexicales exactes (noms de produits, unités)
- BM25 via `ts_rank_cd` compense avec le matching exact
- Pondération 0.7/0.3 validée empiriquement

### Pourquoi cross-encoder reranking ?
- Les bi-encoders (all-MiniLM-L6-v2) sont rapides mais approximatifs
- Le cross-encoder (ms-marco-MiniLM-L-6-v2) compare directement (question, passage) → score fin
- On ne reranke que 20 candidats → latence acceptable (~200ms)

### Pourquoi chunking par sections Markdown ?
- Les fiches techniques sont déjà structurées par sections sémantiques
- 1 chunk = 1 information cohérente (dosage, stockage, applications...)
- Préfixe contextuel pour le rappel : `Produit: X | Section: Y`

---

## Structure du projet

```
AMT-Rose-Blanche/
├── .env                    # Configuration (DB, modèles)
├── config.py               # Lecture de la configuration
├── requirements.txt        # Dépendances Python
├── sql/
│   └── init.sql            # Schéma PostgreSQL + pgvector
├── src/
│   ├── __init__.py
│   ├── db.py               # Connexion + requêtes PostgreSQL
│   ├── chunker.py          # Découpage Markdown intelligent
│   ├── embedder.py         # Modèle d'embeddings
│   └── retriever.py        # 3 stratégies de recherche
├── documents_md/           # Corpus de fiches techniques (.md)
├── ingest.py               # Script d'ingestion
├── search.py               # CLI de recherche
├── api.py                  # API FastAPI
├── evaluate.py             # Benchmark (Recall@3, MRR@3)
├── gold_questions.json     # Questions gold pour évaluation
└── README.md               # Ce fichier
```

---

## Limites et améliorations

### Limites actuelles
- Corpus limité à 35 documents (faible volume)
- Modèle all-MiniLM-L6-v2 : 384 dimensions, anglais-centré (les fiches sont bilingues FR/EN)
- Pas de cache des embeddings de requêtes
- Pas d'authentification sur l'API

### Améliorations possibles
- **Modèle multilingue** : `paraphrase-multilingual-MiniLM-L12-v2` pour mieux gérer le français
- **pgvector** : migration possible avec `VECTOR(384)` + index HNSW pour de grands corpus
- **LLM generation** : ajouter une phase de génération (GPT/Mistral) après le retrieval
- **Cache Redis** : cacher les embeddings de requêtes fréquentes
- **UI Streamlit** : interface utilisateur graphique
- **Feedback loop** : collecter les clics utilisateurs pour affiner le ranking
- **Chunking adaptatif** : ajuster la taille selon la longueur des sections
