# GeoOpKG

GeoOpKG is an intelligent query framework for geoscientific operators. It combines large language models with a graph-based retrieval pipeline (Neo4j), vector search (Weaviate), and a GNN encoder to answer questions about GIS operators and their inputs, outputs, and relationships.

## Highlights

- GraphRAG pipeline integrating Neo4j + Weaviate + LLM
- Intent recognition and operator/entity disambiguation
- GNN-based subgraph encoding for operator similarity and context
- Multi-source operator knowledge base (ArcGIS, QGIS, GEE, etc.)

## Repository Structure

```
GeoOpKG/
	operator_question_answering.py
	preprocessing/
	data/
	LICENSE
	README.md
```

- `operator_question_answering.py`: Main QA pipeline (LLM + Weaviate + Neo4j + GNN)
- `preprocessing/`: Scripts for preparing CSVs and algorithm mappings
- `data/`: Datasets and import scripts for Neo4j, plus sample questions and diagrams

## System Components

1. **Neo4j knowledge graph**: Stores operators, software, parameters, and algorithm relations.
2. **Weaviate vector store**: Hosts vectorized operator descriptions and supports similarity recall.
3. **LLM reasoning**: Classifies intent, rewrites questions, and generates final answers.
4. **GNN encoder (RGCN)**: Builds embeddings from graph substructures to improve ranking.

## Prerequisites

- Python 3.x
- Neo4j (local instance expected at `bolt://localhost:7687`)
- Weaviate (local instance expected at `localhost:8080`, gRPC `50051`)
- A DashScope API key (for Qwen models)

## Python Dependencies

Install core dependencies used by the QA pipeline:

```bash
pip install -r GeoOpKG/requirements.txt
```

For preprocessing and import scripts:

```bash
pip install pandas py2neo
```

Note: `torch-geometric` requires a matching PyTorch build. Install the correct wheels for your CUDA/CPU environment.

## Build the Knowledge Graph (Neo4j)

The knowledge graph is built from CSVs in `data/Import into Neo4j`. Follow the step-by-step instructions in:

- `data/Import into Neo4j/README.md`

This process creates `Software`, `Operation`, `Input`, `Output`, and `Algorithm` nodes and their relationships.

## Prepare Weaviate Collections

The QA script expects the following Weaviate collections to exist and be populated:

- `VectorData` (for entity disambiguation)
- `Prompt_Desc` (for coarse operator recall)

Populate them with vectorized operator names and descriptions consistent with the Neo4j graph.

Use the Weaviate build script to create and populate collections from Neo4j:

```bash
python GeoOpKG/data/build_vector_database.py
```

## Configure the QA Script

Open `operator_question_answering.py` and set the following:

- `dashscope.api_key`: DashScope API key
- `HTTP_PROXY` / `HTTPS_PROXY`: if your environment requires a proxy
- `SentenceTransformer(...)`: set a model name or local path
	- The script uses a 384-dim projection head; use a compatible embedding model or adjust the head
- Neo4j credentials: `uri`, `username`, `password`

## Run the QA Pipeline

From the repository root:

```bash
python GeoOpKG/operator_question_answering.py
```

The script will prompt for a user question and output an answer generated from the graph and vector context.

## Data Assets

- `data/question_90.csv`: Example question set
- `data/operator_ontology.png`: Operator ontology diagram
- `data/Import into Neo4j/`: CSVs and import scripts for Neo4j
- `data/GeoOpKG.zip`: Exported full knowledge-graph snapshot from a built Neo4j instance (optional data snapshot)

## License

See `LICENSE` for usage terms.
