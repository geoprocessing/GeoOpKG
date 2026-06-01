import dashscope
from http import HTTPStatus
from dashscope import Generation
import weaviate
from weaviate import Client
import weaviate.classes as wvc
from weaviate.connect import ConnectionParams
from neo4j import GraphDatabase, exceptions
import os
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv  # Can also be replaced with GATConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F
from collections import defaultdict
from sentence_transformers import CrossEncoder

# Auto-detect GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add after imports and before model definition:
projection_head = nn.Linear(384, 128).to(device)  # Convert 384 dimensions to 128 dimensions

# Please set proxy before running
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# Please set Dashscope API key before running
dashscope.api_key = ""


# Initialize SentenceTransformer model
model = SentenceTransformer('')

# Please set Neo4j connection info
uri = "bolt://localhost:7687"
username = ""
password = ""

# Weaviate connection info
weaviate_client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
)


def call_turing_api(prompt):
    """Call Tongyi Qianwen API"""
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = Generation.call(
            model='qwen3-max-2026-01-23',  # Select model version as needed
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            result = response.output['choices'][0]['message']['content'].strip()
            return result
        else:
            return f"Error: {response.status_code} - {response.message}"
    except Exception as e:
        return f"Error: {e}"

def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        # Avoid NaN
        if str(value).lower() in ('nan', 'inf', '-inf'):
            return ""
        return str(value)
    return str(value)

## Intent Recognition
def classify_intent_with_llm(text):
    """Call Tongyi Qianwen API to extract potential entities and store them in a list"""
    prompt = (
        f"You are an expert in the GIS domain. Based on the following question, determine its intent category.\n"
        f"Available categories:\n"
        f"Known Operator: The question asks about a specific operator, including its parameters, usage, underlying principles, input/output data types, or differences between operators.\n"
        f"Unknown Operator: The question does not mention a specific operator name and aims to find appropriate operators based on a task or operation description.\n"
        f"Similar Operator: The question asks for operators similar to a given operator and their related information.\n"
        f"Note: Only return the category name listed above. Do not provide any explanation.\n"
        f"Pay attention to verbs that partially or fully match operator names, and distinguish whether they are used as operator names or as general verbs.\n"
        f"Example output: Known Operator\n"
        f"Question: {text}"

    )

    result = call_turing_api(prompt)
    intent = result.strip().upper()
    if intent not in ["KNOWN OPERATOR","UNKNOWN OPERATOR","SIMILAR OPERATOR"]:
        result = call_turing_api(prompt)
        intent = result.strip().upper()
    return intent


## Extract Entities
def extract_entities_with_turing(text):
    """Call Tongyi Qianwen API to extract potential entities and store them in a list"""
    prompt = (
        f"You are an expert in the GIS domain. Please extract the GIS operator names from the following user question:\n"
        f"{text}\n\n"
        f"Requirements:\n"
        f"- Only extract operator names explicitly mentioned in the question (e.g., Buffer, Clip, Dissolve). Ignore software names, parameters, and task descriptions.\n"
        f"- If there are spelling errors, automatically correct them to the standard operator names (e.g., 'bufffer' → 'Buffer').\n"
        f"- Operator names must be extracted completely (e.g., ee.Image.reduceRegion, Cost Path).\n"
        f"- Separate operator names using only commas (,). Do not add numbering, explanations, or any extra punctuation.\n"
        f"- If no operator name is mentioned in the question, do not infer or analyze; output exactly: None\n"
    )

    result = call_turing_api(prompt)
    if result:
        # Split the extracted entities and store in a list
        entities = [entity.strip() for entity in result.split(',') if entity.strip()]
        return entities
    return []


## Entity Disambiguation
# Vector query function: find the most similar entities based on input text
def find_similar_entity(query_text,top_k=3):
    query_vector = model.encode(query_text).tolist()

    # Select collection
    questions = weaviate_client.collections.get("VectorData")
    # Vector retrieval
    response = questions.query.near_vector(
        near_vector= query_vector,
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(certainty=True)
    )

    # 4. Parse results (new version returns a different structure than v3)
    results = []
    for obj in response.objects:
        name = obj.properties.get("name", "")
        label = obj.properties.get("label", "")
        certainty = obj.metadata.certainty if obj.metadata else None
        if name:
            results.append((name, label, certainty))

    # Sort by certainty (optional, Weaviate is default sorted)
    results.sort(key=lambda x: x[2] if x[2] is not None else 0, reverse=True)

    # Return top_k, keep only name and label
    return [(res[0], res[1]) for res in results[:top_k]]  # 👈 Return multiple!


def disambiguate_entities(entities):
    """Disambiguate each entity in the entity list"""
    seen=set()
    disambiguated_results = []
    for entity in entities:
        #print(f"entity:",entity)
        # Find the most similar entity
        best_matches = find_similar_entity(entity,top_k=3)
        if not best_matches:
            # If no match, keep original entity (but label is empty)
            fallback = (entity.strip(), "")
            key = (fallback[0].lower(), fallback[1].lower())
            if key not in seen:
                seen.add(key)
                disambiguated_results.append(fallback)
        else:
            # Iterate through top_k results, take the first unseen one
            added = False
            for name, label in best_matches:
                name_clean = name.strip()
                label_clean = label.strip() if label else ""
                key = (name_clean.lower(), label_clean.lower())
                if key not in seen:
                    seen.add(key)
                    disambiguated_results.append((name_clean, label_clean))
                    added = True
                    break  # Only take the first non-repeated high confidence result
            if not added:
                # All top_k are repeated? Skip (or optional fallback)
                pass

    return disambiguated_results


## User Question Interpretation
def rewrite_user_question(text):
    """Rewrite user question into a standard descriptive sentence focusing on underlying GIS operations"""
    prompt = (
        f"You are a professional expert in the GIS domain. Please rewrite the following user question into a single sentence that describes a **specific GIS operation or operator function**.\n"
        f"This rewritten sentence will be used to match GIS tools (e.g., ArcGIS/QGIS operators).\n"
        f"Requirements:\n"
        f"- Focus on **low-level operations** (e.g., calculation, overlay, reclassification, extraction), rather than high-level analytical tasks.\n"
        f"- Use standard GIS terminology (e.g., raster, vector, reprojection, buffer, overlay analysis).\n"
        f"- Do not provide explanations; output only the rewritten sentence, in English.\n"
        f"- If the original question involves specific software, make sure it is explicitly mentioned in the rewritten result.\n\n"
        f"User question: {text}\n"
        f"Rewritten result:"
    )

    result = call_turing_api(prompt)
    if result and not result.startswith("Error"):
        return result.strip()
    else:
        # fallback: return original question directly
        print("⚠️ Rewrite failed, using the original question.")
        return text

## Coarse Filter Operators
def query_prompt_desc_collection(rewritten_question, top_k=30):
    """
    Query the most similar operator names to the rewritten question in Weaviate's Prompt_Desc collection
    Return: [(name1, "Operation"), (name2, "Operation"), ...]
    """
    query_vector = model.encode(rewritten_question).tolist()

    collection = weaviate_client.collections.get("Prompt_Desc")

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(certainty=True)
    )

    results = []
    for obj in response.objects:
        name = obj.properties.get("name", "").strip()
        if name:
            results.append(name)

    # === Disambiguate + Deduplicate ===
    seen = set()
    cleaned_entities = []
    for name in results:
        if name not in seen:
            seen.add(name)
            cleaned_entities.append((name, "Operation"))  # 👈 Change to tuple

    #print(f"🎯 Vector retrieval found {len(results)} raw candidates, after disambiguation kept {len(cleaned_entities)}: {[n for n, _ in cleaned_entities[:5]]}")
    return cleaned_entities  # List[Tuple[str, str]]



# 1. Define GNN model
class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = torch.relu(h)
        h = self.conv2(h, edge_index, edge_type)
        return h


def encode_node_features_complete(node):
    """
    Concatenate heterogeneous node attributes into a unified text for vectorization.
    Supports different nodes having different attributes.
    """
    parts = []

    # 1. Main identifier (Title or Name)
    main_id = node.get("Title") or node.get("Name") or "Unknown"
    parts.append(f"Name: {main_id}")

    # 2. Node type (Extract the first from Labels as type)
    node_type = "Unknown"
    labels = node.get("Labels", [])
    if isinstance(labels, list) and len(labels) > 0:
        node_type = labels[0]
    elif isinstance(labels, str):
        node_type = labels
    parts.append(f"Type: {node_type}")

    # 3. Add common attributes by priority (fixed order)
    for key in ["Description", "DataType", "Version"]:
        val = node.get(key)
        if val and isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")

    # 4. Other attributes (exclude already processed ones)
    exclude_keys = {"Title", "Name", "Description", "DataType", "Version", "Labels"}
    for key, val in node.items():
        if key in exclude_keys:
            continue
        if not val:
            continue
        # Convert to string
        if isinstance(val, (list, dict, set)):
            val = str(val)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")

    # 5. Concatenate + Truncate (SentenceTransformer supports 512 tokens by default, approx 2000 characters)
    text = ". ".join(parts)
    if len(text) > 1500:
        text = text[:1497] + "..."
    return text


# Full graph GNN encoding
def load_full_graph_from_neo4j():
    """Get entire Knowledge Graph"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    nodes, edges, edge_types = [], [], []
    node2id, rel2id = {}, {}
    operation_indices = []  # 👈 Added: store indices of all Operation nodes in the nodes list

    with driver.session() as session:
        # Get nodes
        result = session.run("MATCH (n) RETURN id(n) as id, n as props, labels(n) as labels")
        for record in result:
            nid = record["id"]
            props = dict(record["props"])
            props["Labels"] = record["labels"]
            node2id[nid] = len(nodes)
            nodes.append(props)

            # 👇 Key: check if it is of Operation type (based on labels)
            # Assume the label of Operation node is Operation or contains Operation
            if "Operation" in record["labels"]:
                operation_indices.append(len(nodes))  # Record the position of the node in nodes

            node2id[nid] = len(nodes)
            nodes.append(props)

        # Get edges
        result = session.run("MATCH (n)-[r]->(m) RETURN id(n) as src, id(m) as dst, type(r) as rel")
        for record in result:
            src_id = node2id.get(record["src"])
            dst_id = node2id.get(record["dst"])
            if src_id is not None and dst_id is not None:
                rel_type = record["rel"]
                if rel_type not in rel2id:
                    rel2id[rel_type] = len(rel2id)
                edges.append((src_id, dst_id))
                edge_types.append(rel2id[rel_type])

    driver.close()

    # Build PyG Data
    # Node feature encoding
    texts = [encode_node_features_complete(n) for n in nodes]
    x = torch.tensor(model.encode(texts), dtype=torch.float)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.orig_edge_index = edge_index
    data.orig_edge_type = edge_type
    data.rel2id = rel2id
    data.operation_indices = torch.tensor(operation_indices, dtype=torch.long)  # 👈 Save Operation node index

    rel_num = len(rel2id)
    #print(f"📊 Full graph nodes={len(nodes)}, edges={len(edges)}, rel_types={len(rel2id)}")
    return data, nodes, rel_num



# Subgraph encoding
def load_subgraph_complete(entity_tuples: list, hops=4):
    """
    Load subgraph around seed entities (fix path edge error)
    entity_tuples: [(name, label), ...]
    hops: Subgraph hops
    """
    if not entity_tuples:
        return None, [], 0, [], {}

    driver = GraphDatabase.driver(uri, auth=(username, password))
    nodes, edges, edge_types = [], [], []
    node2id, rel2id = {}, {}
    triples = []

    try:
        with driver.session() as session:
            names = [e[0] for e in entity_tuples]
            lower_titles = [t.lower() for t in names]

            # ✅ Fix: extract each edge in the path
            query = f"""
            MATCH path = (n)-[r*1..{hops}]->(m)
            WHERE any(key IN ['Title','Name'] WHERE n[key] IS NOT NULL AND toLower(n[key]) IN $lower_titles)
                OR any(key IN ['Title','Name'] WHERE m[key] IS NOT NULL AND toLower(m[key]) IN $lower_titles)
            WITH relationships(path) AS rels
            UNWIND rels AS rel
            RETURN startNode(rel) AS src, endNode(rel) AS dst, rel
            LIMIT 5000
            """
            result = session.run(query, titles=names, lower_titles=lower_titles)

            for record in result:
                src_node = record["src"]
                dst_node = record["dst"]
                rel = record["rel"]

                # Source node
                src_props = dict(src_node)
                src_props["Labels"] = list(src_node.labels)
                src_eid = src_node.element_id
                if src_eid not in node2id:
                    node2id[src_eid] = len(nodes)
                    nodes.append(src_props)

                # Target node
                dst_props = dict(dst_node)
                dst_props["Labels"] = list(dst_node.labels)
                dst_eid = dst_node.element_id
                if dst_eid not in node2id:
                    node2id[dst_eid] = len(nodes)
                    nodes.append(dst_props)

                # Edges
                rel_type = rel.type
                if rel_type not in rel2id:
                    rel2id[rel_type] = len(rel2id)
                edges.append((node2id[src_eid], node2id[dst_eid]))
                edge_types.append(rel2id[rel_type])
                triples.append((
                    src_props.get("Title", src_props.get("Name", "")),
                    rel_type,
                    dst_props.get("Title", dst_props.get("Name", ""))
                ))

    finally:
        driver.close()

    # Node feature encoding
    texts = [encode_node_features_complete(n) for n in nodes]
    x = torch.tensor(model.encode(texts), dtype=torch.float).to(device)

    # Original directed edge, used for GNN forward
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)

    # Build Data (remove unused bidirectional edges to avoid confusion)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.orig_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data.orig_edge_type = torch.tensor(edge_types, dtype=torch.long)
    data.rel2id = rel2id

    rel_num = len(rel2id)
    #print(f"📊 Subgraph nodes={len(nodes)}, edges={len(edges)}, rel_types={rel_num}")

    return data, nodes, rel_num, triples, rel2id

def load_undirected_subgraph(entity_tuples: list, hops=4):
    """
    Designed for method 3: extract undirected subgraph to find other operators under the same algorithm category.
    """
    if not entity_tuples:
        return None, [], 0, [], {}

    driver = GraphDatabase.driver(uri, auth=(username, password))
    nodes, edges, edge_types = [], [], []
    node2id, rel2id = {}, {}
    triples = []

    try:
        with driver.session() as session:
            names = [e[0] for e in entity_tuples]

            # ✅ Key change: use undirected path -(r*1..{hops})-
            query = f"""
            MATCH (n)
            WHERE any(key IN ['Title','Name'] WHERE n[key] IN $titles)
            MATCH path = (n)-[*1..{hops}]-(m)
            UNWIND relationships(path) AS rel
            RETURN startNode(rel) AS src, endNode(rel) AS dst, rel
            LIMIT 20000
            """
            result = session.run(query, titles=names)

            for record in result:
                src_node = record["src"]
                dst_node = record["dst"]
                rel = record["rel"]

                src_props = dict(src_node)
                src_props["Labels"] = list(src_node.labels)
                src_eid = src_node.element_id
                if src_eid not in node2id:
                    node2id[src_eid] = len(nodes)
                    nodes.append(src_props)

                dst_props = dict(dst_node)
                dst_props["Labels"] = list(dst_node.labels)
                dst_eid = dst_node.element_id
                if dst_eid not in node2id:
                    node2id[dst_eid] = len(nodes)
                    nodes.append(dst_props)

                rel_type = rel.type
                if rel_type not in rel2id:
                    rel2id[rel_type] = len(rel2id)
                edges.append((node2id[src_eid], node2id[dst_eid]))
                edge_types.append(rel2id[rel_type])
                triples.append((
                    src_props.get("Title", src_props.get("Name", "")),
                    rel_type,
                    dst_props.get("Title", dst_props.get("Name", ""))
                ))

    finally:
        driver.close()

    # Encode node features (consistent with original function)
    texts = [encode_node_features_complete(n) for n in nodes]
    x = torch.tensor(model.encode(texts), dtype=torch.float).to(device)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.orig_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data.orig_edge_type = torch.tensor(edge_types, dtype=torch.long)
    data.rel2id = rel2id

    rel_num = len(rel2id)
    #print(f"📊 Undirected subgraph nodes={len(nodes)}, edges={len(edges)}, rel_types={rel_num}")
    return data, nodes, rel_num, triples, rel2id

def build_edges_with_relation_names(edge_index, edge_type, rel2id):
    """
    Convert edge_index + edge_type to list of (src, dst, rel_name)
    """
    edge_index = edge_index.cpu().numpy()
    edge_type = edge_type.cpu().numpy()
    id2rel = {v: k for k, v in rel2id.items()}
    edges = []
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        rel_id = int(edge_type[i])
        rel_name = id2rel.get(rel_id, "UNKNOWN_REL")
        edges.append((src, dst, rel_name))
    return edges


def collect_reachable_nodes_with_relations(
    subgraph_data,
    subgraph_nodes,
    seed_idx,          # ← Changed to passing index, rather than name
    max_hops=4
):
    """
    BFS expansion starting from seed_idx
    """
    if seed_idx < 0 or seed_idx >= len(subgraph_nodes):
        return []

    # Build adjacency list (out-edges only)
    edge_index = subgraph_data.orig_edge_index.cpu().numpy()
    edge_type = subgraph_data.orig_edge_type.cpu().numpy()
    id2rel = {v: k for k, v in subgraph_data.rel2id.items()}

    adj = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        rel = id2rel.get(int(edge_type[i]), "UNKNOWN_REL")
        if src not in adj:
            adj[src] = []
        adj[src].append((dst, rel))

    # BFS
    from collections import deque
    queue = deque()
    queue.append((seed_idx, [], [subgraph_nodes[seed_idx]], 0))
    visited = set()
    results = []

    while queue:
        node_idx, path_rels, path_nodes, hops = queue.popleft()

        if hops > 0:
            rel_path_str = " → ".join(path_rels)
            results.append({
                "relation_path": rel_path_str,
                "final_node": subgraph_nodes[node_idx],
                "path_nodes": path_nodes
            })

        if hops >= max_hops:
            continue
        if node_idx in visited:
            continue
        visited.add(node_idx)

        if node_idx in adj:
            for nbr_idx, rel in adj[node_idx]:
                if nbr_idx not in visited:
                    new_rels = path_rels + [rel]
                    new_nodes = path_nodes + [subgraph_nodes[nbr_idx]]
                    queue.append((nbr_idx, new_rels, new_nodes, hops + 1))

    return results


# Known Operator Questions
def method1_readout_subgraph(data: Data, nodes: list, seed_entities: list,gnn_model, hops=4):
    """
    Targeting "Known Operator Questions"
    Method 1: Support completely duplicate named nodes (both Title + Label identical)
    - Use node index idx for unique identification
    - Inject __source__ for downstream distinction
    """
    gnn_model.eval()
    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.edge_index, data.edge_type)

    # 1️⃣ Find all matched nodes (no break, support duplicate names)
    seed_indices = []
    seed_names=[]
    for name, label in seed_entities:
        for idx, node in enumerate(nodes):
            title_match = (node.get("Title") or "").strip().lower() == name.strip().lower()
            name_match  = (node.get("Name")  or "").strip().lower() == name.strip().lower()
            if title_match or name_match:
                seed_indices.append(idx)  # 👈 Use idx to distinguish duplicate names!
                seed_names.append(name)

    if not seed_indices:
        print("⚠️ Seed node not found in the subgraph!")
        return torch.empty(0,256), [], [],[],[]

    # Build adjacency list for GNN readout (multi-hop)
    edge_index_np = data.orig_edge_index.cpu().numpy()
    edge_type_np = data.orig_edge_type.cpu().numpy()
    neighbors_dict = {i: set() for i in range(len(nodes))}
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        rel_type = edge_type_np[i]
        neighbors_dict[src].add((dst, rel_type))

    # 3️⃣ Directed expansion for each seed node
    candidates_embeddings = []
    candidates_nodes = []
    all_reachable_info_list = []  # Added

    for idx in seed_indices:
        visited = set([idx])
        current_level = {idx}
        for _ in range(hops):
            next_level = set()
            for node_idx in current_level:
                for nbr, rel_type in neighbors_dict.get(node_idx, []):
                    if nbr not in visited:
                        next_level.add(nbr)
            visited.update(next_level)
            current_level = next_level

        neighbor_indices = list(visited - {idx})
        neighbor_emb = node_embeddings[neighbor_indices].mean(dim=0) if neighbor_indices else torch.zeros_like(node_embeddings[0])

        final_emb = torch.cat([node_embeddings[idx], neighbor_emb], dim=0)
        candidates_embeddings.append(final_emb)

        # 👇 Inject distinguishing info
        node_copy = {**nodes[idx]}
        node_copy["__source__"] = (
            node_copy.get("Software") or
            node_copy.get("Name") or
            node_copy.get("Title") or
            node_copy.get("DataType") or
            "Unknown"
        )
        candidates_nodes.append(node_copy)
        # ✅ Collect all reachable nodes within 4 directed hops (with paths)
        reachable_info = collect_reachable_nodes_with_relations(data, nodes, idx, max_hops=4)
        all_reachable_info_list.append(reachable_info)
    return candidates_embeddings, candidates_nodes, [], all_reachable_info_list, []


# Global load cross-encoder (load only once)
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        #print("🔄 Loading Cross-Encoder model...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder


# Unknown Operator Question
def method2_rerank_by_cross_gnn(seed_entities: list, rewritten_question: str, top_k: int = 10, subgraph_hops: int = 4):
    """
    Load large subgraph at once to ensure all operators with the same name are included
    - Use Cross-Encoder for fine ranking (replace gated fusion)
    - Input format is consistent with the first block code
    """
    if not seed_entities:
        return []

    # 1. Extract unique operator names
    unique_names = list({name for name, _ in seed_entities})
    unique_names_lower = {name.strip().lower() for name in unique_names}  # 👈 Lowercase set
    #print("Lowercase set",unique_names_lower)


    # 2. Load large subgraph containing all candidates at once
    entity_tuples = [(name, "Operation") for name in unique_names]
    try:
        data, nodes, rel_num, _, rel2id = load_subgraph_complete(entity_tuples, hops=subgraph_hops)
        if data is None or len(nodes) == 0:
            return []
    except Exception as e:
        print(f"⚠️ Failed to load the large subgraph: {e}")
        return []

    # 3. GNN encoding (global, can assist generating descriptions)
    gnn_model = RGCNEncoder(
        in_dim=data.x.size(1),
        hidden_dim=256,
        out_dim=128,
        num_relations=rel_num
    ).to(device)
    gnn_model.eval()
    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.edge_index, data.edge_type)

    # 4. Build adjacency list (with relation types, for debugging)
    edge_index_np = data.orig_edge_index.cpu().numpy()
    edge_type_np = data.orig_edge_type.cpu().numpy()
    id2rel = {v: k for k, v in rel2id.items()}
    adj_out = defaultdict(list)
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        rel_id = edge_type_np[i]
        rel_name = id2rel.get(rel_id, "UNKNOWN")
        adj_out[src].append((dst, rel_name))

    # 5. Collect all Operation node instances (including duplicates)
    candidates = []  # Each element: {"op_name": str, "desc_text": str, "node": dict}



    for idx, node in enumerate(nodes):
        labels = node.get("Labels", [])
        if isinstance(labels, str):
            labels = [labels]
        if "Operation" not in labels:
            continue
        node_name = node.get("Title").strip()
        if not node_name:
            continue
        if node_name.lower() not in unique_names_lower:
            continue

        # === Generate Natural Language Description (Key: Each instance is independent) ===
        base_desc = encode_node_features_complete(node)

        # === Dynamically Collect Associated Information ===
        software_names = []

        for nbr_idx, rel_name in adj_out.get(idx, []):
            nbr_node = nodes[nbr_idx]
            nbr_labels = nbr_node.get("Labels", [])
            if isinstance(nbr_labels, str):
                nbr_labels = [nbr_labels]

            # Collect Software Info
            elif rel_name == "ImplementedIn" and "Software" in nbr_labels:
                soft_name = nbr_node.get("Name") or nbr_node.get("Title") or "Unknown"
                software_names.append(soft_name)

        # === Concatenate Enhanced Description ===
        enhanced_parts = [base_desc]
        if software_names:
            enhanced_parts.append("Implemented in: " + ", ".join(software_names))

        desc_text = ". ".join(enhanced_parts)

        # Debug print
        # print(f"🔍 Cross-Encoder candidates: Op='{node_name}', ID={idx}, Software={software_names}")

        candidates.append({
            "op_name": node_name,
            "desc_text": desc_text,
            "node": node
        })

    if not candidates:
        return []

    # 6. Cross-Encoder Fine-Ranking
    cross_encoder = get_cross_encoder()
    pairs = [(rewritten_question, cand["desc_text"]) for cand in candidates]
    try:
        cross_scores = cross_encoder.predict(pairs)
    except Exception as e:
        print(f"⚠️ Cross-Encoder scoring failed; falling back to random ranking. {e}")
        import numpy as np
        cross_scores = np.random.rand(len(candidates))

    # Bind scores
    for i, score in enumerate(cross_scores):
        candidates[i]["similarity"] = float(score)

    # 7. Group by operator name, keep the highest score instance
    best_per_op = defaultdict(lambda: {"score": -1e9, "candidate": None})
    for cand in candidates:
        op_name = cand["op_name"]
        if cand["similarity"] > best_per_op[op_name]["score"]:
            best_per_op[op_name] = {"score": cand["similarity"], "candidate": cand}

    # 8. Sort and return top-k
    sorted_items = sorted(best_per_op.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    result_entities = [(item["candidate"]["op_name"], "Operation") for item in sorted_items]

    #print(f"🎯 Top-{len(result_entities)} after Cross-Encoder sorting: {[name for name, _ in result_entities]}")
    return result_entities

# Similar Operator Questions
def method3_topk_candidates_with_neighbors(data: Data, nodes: list, seed_entities: list, gnn_model, top_k=12):
    """
    Method 3 (with fine-ranking):
    1. Get candidate operators via hasPlan → Algorithm
    2. Calculate similarity with seed using GNN embedding
    3. Return top_k most similar candidate entities [(name, label), ...]
    """
    from collections import defaultdict
    import torch.nn.functional as F

    gnn_model.eval()
    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.edge_index, data.edge_type)

    # 1. Find seed nodes
    seed_indices = []
    for name, label in seed_entities:
        name_clean = name.strip().lower()
        for idx, node in enumerate(nodes):
            title = (node.get("Title") or "").strip().lower()
            node_name = (node.get("Name") or "").strip().lower()
            if title == name_clean or node_name == name_clean:
                seed_indices.append(idx)

    if not seed_indices:
        print("⚠️ Seed operator not found.")
        return []

    # 2. Build adjacency list and Algorithm mapping
    edge_index_np = data.orig_edge_index.cpu().numpy()
    edge_type_np = data.orig_edge_type.cpu().numpy()
    id2rel = {v: k for k, v in data.rel2id.items()}
    adj_out = defaultdict(list)
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        rel = id2rel.get(int(edge_type_np[i]), "")
        adj_out[src].append((dst, rel))

    algorithm_to_ops = defaultdict(list)
    op_to_algorithms = defaultdict(list)
    for idx, node in enumerate(nodes):
        labels = node.get("Labels", [])
        if isinstance(labels, str):
            labels = [labels]
        if "Operation" not in labels:
            continue
        for nbr_idx, rel in adj_out.get(idx, []):
            if rel == "hasPlan":  # ← Ensure relationship name is correct
                algorithm_to_ops[nbr_idx].append(idx)
                op_to_algorithms[idx].append(nbr_idx)

    # 3. Collect all candidates (avoid duplicates)
    candidate_set = set()
    for seed_idx in seed_indices:
        seed_algos = op_to_algorithms.get(seed_idx, [])
        for algo_idx in seed_algos:
            for other_op_idx in algorithm_to_ops[algo_idx]:
                if other_op_idx != seed_idx:
                    candidate_set.add(other_op_idx)

    if not candidate_set:
        print("⚠️ No operators of the same category found.")
        return []

    #print(f"🔍 Found {len(candidate_set)} candidate operators, calculating similarity...")

    # 4. Calculate similarity between each seed and all candidates (take max)
    candidate_scores = defaultdict(float)  # cand_idx -> max_similarity
    for seed_idx in seed_indices:
        seed_emb = node_embeddings[seed_idx].unsqueeze(0)  # [1, D]
        candidate_indices = list(candidate_set)
        candidate_embs = node_embeddings[candidate_indices]  # [M, D]
        sims = F.cosine_similarity(seed_emb, candidate_embs)  # [M]

        for i, cand_idx in enumerate(candidate_indices):
            # Keep highest similarity (for multiple seeds)
            candidate_scores[cand_idx] = max(candidate_scores[cand_idx], sims[i].item())

    # 5. Sort by similarity, take top_k
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # 6. Convert to entity list [(name, label), ...]
    seen_entities = set()
    result_entities = []
    for cand_idx, sim_score in sorted_candidates:
        node = nodes[cand_idx]
        name = (node.get("Name") or node.get("Title") or "").strip()
        label="Operation"
        if not name:
            continue
        entity_key=(name, label)
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            result_entities.append(entity_key)


    #print(f"🎯 Top-{len(result_entities)} candidates after fine-ranking: {result_entities}")
    return result_entities


def process_text(text):
    # ========================
    # 1️⃣ Extract intent (with retry and fallback)
    # ========================
    intent = classify_intent_with_llm(text)
    #print("🔍 Initial intent recognition result:", intent)

    VALID_INTENTS = {"KNOWN OPERATOR", "UNKNOWN OPERATOR", "SIMILAR OPERATOR"}
    if intent not in VALID_INTENTS:
        #print("⚠️ Intent recognition invalid, retrying once...")
        intent = classify_intent_with_llm(text)
        if intent not in VALID_INTENTS:
            intent = "KNOWN OPERATOR"
            print(f"❌ Still invalid; using the default intent. {intent}")
        else:
            print("✅ Retry successful; intent:", intent)
    else:
        print("✅ Intent recognition successful:", intent)

    # ========================
    # 2️⃣ Branch processing based on intent
    # ========================
    try:
        if intent == "KNOWN OPERATOR":
            #print("📌 Processing [Known Operator]: extracting and disambiguating entities...")
            entities = extract_entities_with_turing(text)
            #print("Extracted entities list:", entities)
            disambiguated_entities = disambiguate_entities(entities)
            #print("Disambiguated entities list:", disambiguated_entities)
            entity_tuples = [
                (e[0].strip(), e[1]) for e in disambiguated_entities
                if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0].strip()
            ]
            if not entity_tuples:
                raise ValueError("No valid entities extracted.")

            # Load subgraph and execute method 1
            data, nodes, rel_num, triples, rel2id = load_subgraph_complete(entity_tuples, hops=2)
            gnn_model = RGCNEncoder(in_dim=data.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num).to(
                device)
            result = method1_readout_subgraph(data, nodes, entity_tuples, gnn_model, hops=4)
            candidates_embeddings, candidates_nodes, _, all_reachable_info, _ = result

        elif intent == "SIMILAR OPERATOR":
            #print("📌 Processing [Similar Operator]: extracting and disambiguating entities...")
            entities = extract_entities_with_turing(text)
            #print("Extracted entities list:", entities)
            disambiguated_entities = disambiguate_entities(entities)
            #print("Disambiguated entities list:", disambiguated_entities)
            entity_tuples = [
                (e[0].strip(), e[1]) for e in disambiguated_entities
                if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0].strip()
            ]
            if not entity_tuples:
                raise ValueError("No valid entities extracted.")

            # Load subgraph and execute method 3
            data_nudir, nodes_nudir, rel_num_nudir, _, _ = load_undirected_subgraph(entity_tuples, hops=2)
            gnnmodel= RGCNEncoder(in_dim=data_nudir.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num_nudir).to(
                device)
            candidates_entities = method3_topk_candidates_with_neighbors(data_nudir, nodes_nudir, entity_tuples,gnnmodel, top_k=12)

            data, nodes, rel_num, triples, rel2id = load_subgraph_complete(candidates_entities, hops=2)
            gnn_model = RGCNEncoder(in_dim=data.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num).to(
                device)
            result = method1_readout_subgraph(data, nodes, candidates_entities, gnn_model, hops=4)
            candidates_embeddings, candidates_nodes, _, all_reachable_info, _ = result

        elif intent == "UNKNOWN OPERATOR":
            #print("\n📌 Processing [Unknown Operator]: question rewriting + vector retrieval...")
            rewritten = rewrite_user_question(text)
            #print("Interpreted user question:", rewritten)
            seed_entities = query_prompt_desc_collection(rewritten, top_k=30)
            #print("Retrieved operators",seed_entities)
            if not seed_entities:
                raise ValueError("No candidate operators obtained.")

            # Call new method 2 (full graph no longer needed)
            candidates_entities = method2_rerank_by_cross_gnn(seed_entities, rewritten, top_k=10, subgraph_hops=2)

            data, nodes, rel_num, triples, rel2id = load_subgraph_complete(candidates_entities, hops=2)
            gnn_model = RGCNEncoder(in_dim=data.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num).to(
                device)
            result = method1_readout_subgraph(data, nodes, candidates_entities, gnn_model, hops=4)
            candidates_embeddings, candidates_nodes, _, all_reachable_info, _ = result

            if len(candidates_nodes) == 0:
                raise ValueError("No valid operators after GNN re-ranking.")

        else:
            raise RuntimeError(f"*Unhandled intent: {intent}")

    except Exception as e:
        raise RuntimeError(f"Error occurred during the graph processing stage: {str(e)}")

    # ========================
    # 3️⃣ Check candidate results
    # ========================
    if len(candidates_nodes) == 0:
        return "No relevant operator information found; please try a more specific description."

    # print("🔍 Final candidate nodes:")
    for n in candidates_nodes:
        title = n.get('Title', n.get('Name', 'DataType'))
        source = n.get('__source__', 'Unknown')
        labels = ", ".join(n.get('Labels', [])) if isinstance(n.get('Labels'), list) else n.get('Labels', 'Unknown')
        # print(f"  - {title} (Source: {source}, Labels: {labels})")

    # ========================
    # 4️⃣ Build prompt and generate answer
    # ========================
    combined_prompt_parts = ["You are an expert in the GIS domain. Provide detailed and professional answers. Please answer the question based on the following candidate operators and their associated information:\n"]

    for idx, node in enumerate(candidates_nodes):
        title = node.get('Title', node.get('Name', 'DataType'))
        source = node.get('__source__', 'Unknown')
        desc = node.get('Description', 'No description')

        info_parts = [f"operator {idx + 1}: {title} (source: {source})"]
        if desc != 'No description':
            info_parts.append(f"description: {desc}")

        reachable_info = all_reachable_info[idx]

        if reachable_info:
            info_parts.append("  Associated information:")
            seen_generic = set()  # Deduplicate normal edges
            seen_datatype = set()  # Deduplicate hasDataType relations
            for item in reachable_info:
                rel_path = item["relation_path"]
                final_node = item["final_node"]
                path_nodes = item["path_nodes"]

                # Get end node name and type
                n_title = (final_node.get('Title')or final_node.get('Name')or final_node.get('DataType'))
                n_labels = final_node.get('Labels', [])
                n_type = n_labels[0] if isinstance(n_labels, list) and n_labels else str(n_labels)

                # ✅ Specially handle hasDataType: show which parameter it belongs to
                if "hasDataType" in rel_path and len(path_nodes) >= 3:
                    # path_nodes[-2] is the parameter node (e.g., INPUT_RASTER)
                    param_node = path_nodes[-2]
                    param_name = param_node.get('Title', param_node.get('Name', 'UnknownParam'))
                    # Unique key: parameter name
                    if param_name in seen_datatype:
                        continue
                    seen_datatype.add(param_name)

                    info_parts.append(
                        f"  - Parameter '{param_name}' Data type: {n_title} ({n_type}) "
                        f"[{final_node.get('DataType', 'N/A')}]"
                    )
                else:
                    # Normal relation: deduplicate by (path, node name)
                    unique_key = (rel_path, n_title)
                    if unique_key in seen_generic:
                        continue
                    seen_generic.add(unique_key)

                    # Normal relation
                    extra_info = []
                    if final_node.get('Description'):
                        extra_info.append(f"Description: {final_node['Description']}")
                    if final_node.get('DataType'):
                        extra_info.append(f"DataType: {final_node['DataType']}")

                    extra_str = " | ".join(extra_info)
                    if extra_str:
                        info_parts.append(f"    - {rel_path}: {n_title} ({n_type}) [{extra_str}]")
                    else:
                        info_parts.append(f"    - {rel_path}: {n_title} ({n_type})")

        combined_prompt_parts.extend(info_parts)

    combined_prompt_parts.append(f"\nUser question is: {text}")
    combined_prompt_parts.append(
        f"\n=== Answer Requirements ===\n"
        f"1. First, understand the core intent of the user's question. Select only the **most relevant parts** of the candidate knowledge for analysis, and provide a corresponding answer based on the question. Do not include unrelated information.\n"
        f"2. Prioritize answering operators that exactly match the user's question (case-insensitive). If there are operators with the same name, explain their functions and parameters according to the software classification.\n"
        f"3. If information is missing, you may provide appropriate supplementation, but do not fabricate information.\n"
        f"4. Use professional yet easy-to-understand terminology, avoiding IDs, internal identifiers, or other information irrelevant to the user.\n"
        f"Please output the answer directly without including analysis phrases such as 'based on the query results'."
        )
    final_prompt = "\n".join(combined_prompt_parts)

    # Call LLM
    try:
        answer = call_turing_api(final_prompt)
        if answer.startswith("Error"):
            raise RuntimeError(f"LLM call failed: {answer}")
        return answer
    except Exception as e:
        return f"Error occurred while generating the answer: {str(e)}"


# Main Program
if __name__ == '__main__':
    try:
        text = input("Please enter the text content: ")
        result = process_text(text)
        print("result:", result)
    finally:
        weaviate_client.close()  # Ensure it is closed
        #print("✅ Weaviate connection has been closed.")





