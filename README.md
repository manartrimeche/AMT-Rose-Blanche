# Module de recherche sémantique (RAG)

Ce projet fournit un module Python pour interroger un corpus PDF via recherche sémantique.

## Fonctionnalités

- Reçoit une question utilisateur en langage naturel
- Génère son embedding sémantique
- Compare la question aux fragments indexés (similarité cosinus)
- Classe les résultats par pertinence décroissante
- Retourne les 3 fragments les plus pertinents avec:
  - Texte du fragment
  - Score de similarité

## Dataset utilisé

Le module est configuré par défaut pour utiliser:

- `data/enzymes/*.md` (fiches techniques Markdown structurées par sections)

## Installation

```bash
pip install -r requirements.txt
```

## 1) Construire la base vectorielle

```bash
python rag_semantic_search.py index --data-dir data/enzymes --output-dir vector_store --model all-MiniLM-L6-v2
```

Fichiers générés dans `vector_store/`:

- `embeddings.npy`
- `chunks.jsonl`
- `metadata.json`

## 2) Interroger la base vectorielle

```bash
python rag_semantic_search.py query --output-dir vector_store --model all-MiniLM-L6-v2 --question "Quelle enzyme est adaptée à l'amélioration du volume ?"
```

Le module renvoie les 3 fragments les plus pertinents avec le score de similarité cosinus.

## Paramètres utiles

- `--chunk-size` : taille des fragments en mots (défaut: `180`)
- `--chunk-overlap` : chevauchement en mots (défaut: `30`)
- `--top-k` : nombre de résultats (défaut: `3`)
- `--model` : modèle d'embedding sentence-transformers
