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
from torch_geometric.nn import RGCNConv  # 也可以换成 GATConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F
from collections import defaultdict
from sentence_transformers import CrossEncoder

# 自动检测 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在 import 之后、模型定义之前添加：
projection_head = nn.Linear(384, 128).to(device)  # 将 384 维转为 128 维

# 请在运行前设置代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 请在运行前设置 Dashscope API 密钥
dashscope.api_key = ""


# 初始化 SentenceTransformer 模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 请设置Neo4j 连接信息
uri = "bolt://localhost:7687"
username = ""
password = ""

# Weaviate 连接信息
weaviate_client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
)


def call_turing_api(prompt):
    """调用通义千问 API"""
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = Generation.call(
            model='qwen3-max-2025-09-23',  # 根据需要选择模型版本
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
        # 避免 NaN
        if str(value).lower() in ('nan', 'inf', '-inf'):
            return ""
        return str(value)
    return str(value)

##意图识别
def classify_intent_with_llm(text):
    """通过调用通义千问 API 提取潜在的实体并存储在列表中"""
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
    if intent not in ["已知算子类别","未知算子类别","相似算子问题"]:
        result=call_turing_api()
        intent=result.strip().upper()
    return intent


##提取实体
def extract_entities_with_turing(text):
    """通过调用通义千问 API 提取潜在的实体并存储在列表中"""
    prompt = (
        f"You are an expert in the GIS domain. Please extract the GIS operator names from the following user question:\n"
        f"{text}\n\n"
        f"Requirements:\n"
        f"- Only extract operator names explicitly mentioned in the question (e.g., Buffer, Clip, Dissolve). Ignore software names, parameters, and task descriptions.\n"
        f"- If there are spelling errors, automatically correct them to the standard operator names (e.g., 'bufffer' → 'Buffer').\n"
        f"- Separate operator names using only commas (,). Do not add numbering, explanations, or any extra punctuation.\n"
        f"- If no operator name is mentioned in the question, do not infer or analyze; output exactly: None\n"
    )

    result = call_turing_api(prompt)
    if result:
        # 将提取的实体按行分割，存入列表
        entities = [entity.strip() for entity in result.split(',') if entity.strip()]
        return entities
    return []


##实体消岐
# 向量查询函数：根据输入文本找到最相似的实体
def find_similar_entity(query_text,top_k=3):
    query_vector = model.encode(query_text).tolist()

    #选择集合
    questions = weaviate_client.collections.get("VectorData")
    #向量检索
    response = questions.query.near_vector(
        near_vector= query_vector,
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(certainty=True)
    )

    # 4. 解析结果（新版返回的结构和 v3 不一样）
    results = []
    for obj in response.objects:
        name = obj.properties.get("name", "")
        label = obj.properties.get("label", "")
        certainty = obj.metadata.certainty if obj.metadata else None
        if name:
            results.append((name, label, certainty))

    # 按 certainty 排序（可选，Weaviate 默认已排序）
    results.sort(key=lambda x: x[2] if x[2] is not None else 0, reverse=True)

    # 返回前 top_k 个，只保留 name 和 label
    return [(res[0], res[1]) for res in results[:top_k]]  # 👈 返回多个！


def disambiguate_entities(entities):
    """消歧实体列表中的每个实体"""
    seen=set()
    disambiguated_results = []
    for entity in entities:
        #print(f"entity:",entity)
        # 找到最相似的实体
        best_matches = find_similar_entity(entity,top_k=3)
        if not best_matches:
            # 如果没有匹配，保留原始实体（但 label 为空）
            fallback = (entity.strip(), "")
            key = (fallback[0].lower(), fallback[1].lower())
            if key not in seen:
                seen.add(key)
                disambiguated_results.append(fallback)
        else:
            # 遍历 top_k 结果，取第一个未见过的
            added = False
            for name, label in best_matches:
                name_clean = name.strip()
                label_clean = label.strip() if label else ""
                key = (name_clean.lower(), label_clean.lower())
                if key not in seen:
                    seen.add(key)
                    disambiguated_results.append((name_clean, label_clean))
                    added = True
                    break  # 只取第一个未重复的高置信结果
            if not added:
                # 所有 top_k 都重复了？那就跳过（或可选 fallback）
                pass

    return disambiguated_results


##用户问题解读
def rewrite_user_question(text):
    """将用户问题改写为聚焦 GIS 底层操作的标准描述句"""
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
        # fallback：直接返回原问题
        print("⚠️ Rewrite failed, using the original question.")
        return text

##粗筛算子
def query_prompt_desc_collection(rewritten_question, top_k=30):
    """
    在 Weaviate 的 Prompt_Desc 集合中查询与改写后问题最相似的算子名称
    返回: [(name1, "Operation"), (name2, "Operation"), ...]
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

    # === 消岐 + 去重 ===
    seen = set()
    cleaned_entities = []
    for name in results:
        if name not in seen:
            seen.add(name)
            cleaned_entities.append((name, "Operation"))  # 👈 改为元组

    #print(f"🎯 向量检索到 {len(results)} 个原始候选，消岐后保留 {len(cleaned_entities)} 个: {[n for n, _ in cleaned_entities[:5]]}")
    return cleaned_entities  # List[Tuple[str, str]]



# 1. 定义 GNN 模型
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
    将异构节点属性拼接为统一文本，用于向量化。
    支持不同节点具有不同属性。
    """
    parts = []

    # 1. 主标识（Title 或 Name）
    main_id = node.get("Title") or node.get("Name") or "Unknown"
    parts.append(f"Name: {main_id}")

    # 2. 节点类型（从 Labels 提取第一个作为类型）
    node_type = "Unknown"
    labels = node.get("Labels", [])
    if isinstance(labels, list) and len(labels) > 0:
        node_type = labels[0]
    elif isinstance(labels, str):
        node_type = labels
    parts.append(f"Type: {node_type}")

    # 3. 按优先级添加常用属性（固定顺序）
    for key in ["Description", "DataType", "Version"]:
        val = node.get(key)
        if val and isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")

    # 4. 其他属性（排除已处理的）
    exclude_keys = {"Title", "Name", "Description", "DataType", "Version", "Labels"}
    for key, val in node.items():
        if key in exclude_keys:
            continue
        if not val:
            continue
        # 转为字符串
        if isinstance(val, (list, dict, set)):
            val = str(val)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")

    # 5. 拼接 + 截断（SentenceTransformer 默认支持512 tokens，约2000字符）
    text = ". ".join(parts)
    if len(text) > 1500:
        text = text[:1497] + "..."
    return text


#全图GNN编码
def load_full_graph_from_neo4j():
    """获取整个知识图谱"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    nodes, edges, edge_types = [], [], []
    node2id, rel2id = {}, {}
    operation_indices = []  # 👈 新增：存储所有 Operation 节点在 nodes 列表中的索引

    with driver.session() as session:
        # 获取节点
        result = session.run("MATCH (n) RETURN id(n) as id, n as props, labels(n) as labels")
        for record in result:
            nid = record["id"]
            props = dict(record["props"])
            props["Labels"] = record["labels"]
            node2id[nid] = len(nodes)
            nodes.append(props)

            # 👇 关键：判断是否为 Operation 类型（根据标签）
            # 假设 Operation 节点的 label 是 'Operation' 或包含 'Operation'
            if "Operation" in record["labels"]:
                operation_indices.append(len(nodes))  # 记录该节点在 nodes 中的位置

            node2id[nid] = len(nodes)
            nodes.append(props)

        # 获取边
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

    # 构建 PyG Data
    # 节点特征编码
    texts = [encode_node_features_complete(n) for n in nodes]
    x = torch.tensor(model.encode(texts), dtype=torch.float)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.orig_edge_index = edge_index
    data.orig_edge_type = edge_type
    data.rel2id = rel2id
    data.operation_indices = torch.tensor(operation_indices, dtype=torch.long)  # 👈 保存 Operation 节点索引

    rel_num = len(rel2id)
    #print(f"📊 全图节点数={len(nodes)}, 边数={len(edges)}, 关系类型数={len(rel2id)}")
    return data, nodes, rel_num



#子图编码
def load_subgraph_complete(entity_tuples: list, hops=4):
    """
    加载围绕种子实体的子图（修复路径边错误）
    entity_tuples: [(name, label), ...]
    hops: 子图跳数
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

            # ✅ 修复：拆解路径中的每一条边
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

                # 源节点
                src_props = dict(src_node)
                src_props["Labels"] = list(src_node.labels)
                src_eid = src_node.element_id
                if src_eid not in node2id:
                    node2id[src_eid] = len(nodes)
                    nodes.append(src_props)

                # 目标节点
                dst_props = dict(dst_node)
                dst_props["Labels"] = list(dst_node.labels)
                dst_eid = dst_node.element_id
                if dst_eid not in node2id:
                    node2id[dst_eid] = len(nodes)
                    nodes.append(dst_props)

                # 边
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

    # 节点特征编码
    texts = [encode_node_features_complete(n) for n in nodes]
    x = torch.tensor(model.encode(texts), dtype=torch.float).to(device)

    # 原始有向边，用于 GNN forward
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)

    # 构造 Data（移除未使用的双向边，避免混淆）
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.orig_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data.orig_edge_type = torch.tensor(edge_types, dtype=torch.long)
    data.rel2id = rel2id

    rel_num = len(rel2id)
    #print(f"📊 子图节点数={len(nodes)}, 边数={len(edges)}, 关系类型数={rel_num}")

    return data, nodes, rel_num, triples, rel2id

def load_undirected_subgraph(entity_tuples: list, hops=4):
    """
    专为方法三设计：加载无向子图，用于查找同算法类别下的其他算子。
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

            # ✅ 关键修改：使用无向路径 -(r*1..{hops})-
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

    # 编码节点特征（与原函数一致）
    texts = [encode_node_features_complete(n) for n in nodes]
    x = torch.tensor(model.encode(texts), dtype=torch.float).to(device)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.orig_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data.orig_edge_type = torch.tensor(edge_types, dtype=torch.long)
    data.rel2id = rel2id

    rel_num = len(rel2id)
    #print(f"📊 无向子图节点数={len(nodes)}, 边数={len(edges)}, 关系类型数={rel_num}")
    return data, nodes, rel_num, triples, rel2id

def build_edges_with_relation_names(edge_index, edge_type, rel2id):
    """
    将 edge_index + edge_type 转为 (src, dst, rel_name) 列表
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
    seed_idx,          # ← 改为传入索引，而非名称
    max_hops=4
):
    """
    从 seed_idx 开始 BFS 扩展
    """
    if seed_idx < 0 or seed_idx >= len(subgraph_nodes):
        return []

    # 构建邻接表（只出边）
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


#已知算子问题
def method1_readout_subgraph(data: Data, nodes: list, seed_entities: list,gnn_model, hops=4):
    """
    针对“已知算子问题”
    方法一：支持完全重名节点（Title + Label 都相同）
    - 用节点索引 idx 唯一标识
    - 注入 __source__ 便于下游区分
    """
    gnn_model.eval()
    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.orig_edge_index, data.orig_edge_type)

    # 1️⃣ 找所有匹配节点（不 break，支持完全重名）
    seed_indices = []
    seed_names=[]
    for name, label in seed_entities:
        for idx, node in enumerate(nodes):
            title_match = (node.get("Title") or "").strip().lower() == name.strip().lower()
            name_match  = (node.get("Name")  or "").strip().lower() == name.strip().lower()
            if title_match or name_match:
                seed_indices.append(idx)  # 👈 用 idx 区分重名节点！
                seed_names.append(name)

    if not seed_indices:
        print("⚠️ Seed node not found in the subgraph!")
        return torch.empty(0,256), [], [],[],[]

    # 构建邻接表用于 GNN readout（多跳）
    edge_index_np = data.orig_edge_index.cpu().numpy()
    edge_type_np = data.orig_edge_type.cpu().numpy()
    neighbors_dict = {i: set() for i in range(len(nodes))}
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        rel_type = edge_type_np[i]
        neighbors_dict[src].add((dst, rel_type))

    # 3️⃣ 对每个 seed 节点进行有向扩展
    candidates_embeddings = []
    candidates_nodes = []
    all_reachable_info_list = []  # 新增

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

        # 👇 注入区分信息
        node_copy = {**nodes[idx]}
        node_copy["__source__"] = (
            node_copy.get("Software") or
            node_copy.get("Name") or
            node_copy.get("Title") or
            node_copy.get("DataType") or
            "Unknown"
        )
        candidates_nodes.append(node_copy)
        # ✅ 收集有向4跳内所有可达节点（带路径）
        reachable_info = collect_reachable_nodes_with_relations(data, nodes, idx, max_hops=4)
        all_reachable_info_list.append(reachable_info)
    return candidates_embeddings, candidates_nodes, [], all_reachable_info_list, []


# 全局加载 cross-encoder（只加载一次）
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        #print("🔄 加载 Cross-Encoder 模型...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder


#未知算子问题
def method2_rerank_by_cross_gnn(seed_entities: list, rewritten_question: str, top_k: int = 10, subgraph_hops: int = 4):
    """
    一次性加载大子图，确保所有同名算子都被包含
    - 使用 Cross-Encoder 精排（替代门控融合）
    - 输入格式与第一段代码一致：seed_entities = [(name, "Operation"), ...]
    """
    if not seed_entities:
        return []

    # 1. 提取唯一算子名称
    unique_names = list({name for name, _ in seed_entities})
    unique_names_lower = {name.strip().lower() for name in unique_names}  # 👈 小写集合
    #print("小写集合",unique_names_lower)


    # 2. 一次性加载包含所有候选的大子图
    entity_tuples = [(name, "Operation") for name in unique_names]
    try:
        data, nodes, rel_num, _, rel2id = load_subgraph_complete(entity_tuples, hops=subgraph_hops)
        if data is None or len(nodes) == 0:
            return []
    except Exception as e:
        print(f"⚠️ Failed to load the large subgraph: {e}")
        return []

    # 3. GNN 编码（全局，可用于辅助生成描述，或保留流程一致性）
    gnn_model = RGCNEncoder(
        in_dim=data.x.size(1),
        hidden_dim=256,
        out_dim=128,
        num_relations=rel_num
    ).to(device)
    gnn_model.eval()
    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.orig_edge_index, data.orig_edge_type)

    # 4. 构建邻接表（带关系类型，用于辅助生成描述或调试）
    edge_index_np = data.orig_edge_index.cpu().numpy()
    edge_type_np = data.orig_edge_type.cpu().numpy()
    id2rel = {v: k for k, v in rel2id.items()}
    adj_out = defaultdict(list)
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        rel_id = edge_type_np[i]
        rel_name = id2rel.get(rel_id, "UNKNOWN")
        adj_out[src].append((dst, rel_name))

    # 5. 收集所有 Operation 节点实例（包括同名多个）
    candidates = []  # 每个元素：{"op_name": str, "desc_text": str, "node": dict}



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

        # === 生成自然语言描述（关键：每个实例独立）===
        base_desc = encode_node_features_complete(node)

        # === 动态收集关联信息 ===
        software_names = []

        for nbr_idx, rel_name in adj_out.get(idx, []):
            nbr_node = nodes[nbr_idx]
            nbr_labels = nbr_node.get("Labels", [])
            if isinstance(nbr_labels, str):
                nbr_labels = [nbr_labels]

            # 收集 Software 信息
            elif rel_name == "ImplementedIn" and "Software" in nbr_labels:
                soft_name = nbr_node.get("Name") or nbr_node.get("Title") or "Unknown"
                software_names.append(soft_name)

        # === 拼接增强描述 ===
        enhanced_parts = [base_desc]
        if software_names:
            enhanced_parts.append("Implemented in: " + ", ".join(software_names))

        desc_text = ". ".join(enhanced_parts)

        # 调试打印
        # print(f"🔍 Cross-Encoder 候选: 算子='{node_name}', ID={idx}, 软件={software_names}")

        candidates.append({
            "op_name": node_name,
            "desc_text": desc_text,
            "node": node
        })

    if not candidates:
        return []

    # 6. Cross-Encoder 精排
    cross_encoder = get_cross_encoder()
    pairs = [(rewritten_question, cand["desc_text"]) for cand in candidates]
    try:
        cross_scores = cross_encoder.predict(pairs)
    except Exception as e:
        print(f"⚠️ Cross-Encoder scoring failed; falling back to random ranking. {e}")
        import numpy as np
        cross_scores = np.random.rand(len(candidates))

    # 绑定分数
    for i, score in enumerate(cross_scores):
        candidates[i]["similarity"] = float(score)

    # 7. 按算子名称分组，保留最高分实例
    best_per_op = defaultdict(lambda: {"score": -1e9, "candidate": None})
    for cand in candidates:
        op_name = cand["op_name"]
        if cand["similarity"] > best_per_op[op_name]["score"]:
            best_per_op[op_name] = {"score": cand["similarity"], "candidate": cand}

    # 8. 排序并返回 top-k
    sorted_items = sorted(best_per_op.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    result_entities = [(item["candidate"]["op_name"], "Operation") for item in sorted_items]

    #print(f"🎯 Cross-Encoder 精排后 Top-{len(result_entities)} 算子: {[name for name, _ in result_entities]}")
    return result_entities

#相似算子问题
def method3_topk_candidates_with_neighbors(data: Data, nodes: list, seed_entities: list, gnn_model, top_k=12):
    """
    方法三（带精排）：
    1. 通过 hasPlan → Algorithm 获取候选算子
    2. 用 GNN 嵌入计算与 seed 的相似度
    3. 返回 top_k 个最相似的候选实体列表 [(name, label), ...]
    """
    from collections import defaultdict
    import torch.nn.functional as F

    gnn_model.eval()
    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.orig_edge_index, data.orig_edge_type)

    # 1. 找 seed 节点
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

    # 2. 构建邻接表和 Algorithm 映射
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
            if rel == "hasPlan":  # ← 确保关系名正确
                algorithm_to_ops[nbr_idx].append(idx)
                op_to_algorithms[idx].append(nbr_idx)

    # 3. 收集所有候选（避免重复）
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

    #print(f"🔍 找到 {len(candidate_set)} 个候选算子，正在计算相似度...")

    # 4. 计算每个 seed 与所有候选的相似度（取最大值）
    candidate_scores = defaultdict(float)  # cand_idx -> max_similarity
    for seed_idx in seed_indices:
        seed_emb = node_embeddings[seed_idx].unsqueeze(0)  # [1, D]
        candidate_indices = list(candidate_set)
        candidate_embs = node_embeddings[candidate_indices]  # [M, D]
        sims = F.cosine_similarity(seed_emb, candidate_embs)  # [M]

        for i, cand_idx in enumerate(candidate_indices):
            # 保留最高相似度（多个 seed 时）
            candidate_scores[cand_idx] = max(candidate_scores[cand_idx], sims[i].item())

    # 5. 按相似度排序，取 top_k
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # 6. 转为实体列表 [(name, label), ...]
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


    #print(f"🎯 精排后 Top-{len(result_entities)} 候选: {result_entities}")
    return result_entities


def process_text(text):
    # ========================
    # 1️⃣ 提取意图（带重试和 fallback）
    # ========================
    intent = classify_intent_with_llm(text)
    #print("🔍 初始意图识别结果:", intent)

    VALID_INTENTS = {"Known Operator", "Unknown Operator", "Similar Operator"}
    if intent not in VALID_INTENTS:
        #print("⚠️ 意图识别无效，重试一次...")
        intent = classify_intent_with_llm(text)
        if intent not in VALID_INTENTS:
            intent = "Known Operator"
            print(f"❌ Still invalid; using the default intent. {intent}")
        else:
            print("✅ Retry successful; intent:", intent)
    else:
        print("✅ Intent recognition successful:", intent)

    # ========================
    # 2️⃣ 根据意图分支处理
    # ========================
    try:
        if intent == "Known Operator":
            #print("\n📌 处理「已知算子类别」：提取并消歧实体...")
            entities = extract_entities_with_turing(text)
            #print("提取的实体列表:", entities)
            disambiguated_entities = disambiguate_entities(entities)
            #print("消歧的实体列表:", disambiguated_entities)
            entity_tuples = [
                (e[0].strip(), e[1]) for e in disambiguated_entities
                if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0].strip()
            ]
            if not entity_tuples:
                raise ValueError("No valid entities extracted.")

            # 加载子图并执行方法一
            data, nodes, rel_num, triples, rel2id = load_subgraph_complete(entity_tuples, hops=2)
            gnn_model = RGCNEncoder(in_dim=data.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num).to(
                device)
            result = method1_readout_subgraph(data, nodes, entity_tuples, gnn_model, hops=4)
            candidates_embeddings, candidates_nodes, _, all_reachable_info, _ = result

        elif intent == "Similar Operator":
            #print("\n📌 处理「相似算子问题」：提取并消歧实体...")
            entities = extract_entities_with_turing(text)
            #print("提取的实体列表:", entities)
            disambiguated_entities = disambiguate_entities(entities)
            #print("消歧的实体列表:", disambiguated_entities)
            entity_tuples = [
                (e[0].strip(), e[1]) for e in disambiguated_entities
                if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0].strip()
            ]
            if not entity_tuples:
                raise ValueError("No valid entities extracted.")

            # 加载子图并执行方法三
            data_nudir, nodes_nudir, rel_num_nudir, _, _ = load_undirected_subgraph(entity_tuples, hops=2)
            gnnmodel= RGCNEncoder(in_dim=data_nudir.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num_nudir).to(
                device)
            candidates_entities = method3_topk_candidates_with_neighbors(data_nudir, nodes_nudir, entity_tuples,gnnmodel, top_k=12)

            data, nodes, rel_num, triples, rel2id = load_subgraph_complete(candidates_entities, hops=2)
            gnn_model = RGCNEncoder(in_dim=data.x.size(1), hidden_dim=256, out_dim=128, num_relations=rel_num).to(
                device)
            result = method1_readout_subgraph(data, nodes, candidates_entities, gnn_model, hops=4)
            candidates_embeddings, candidates_nodes, _, all_reachable_info, _ = result

        elif intent == "Unknown Operator":
            #print("\n📌 处理「未知算子类别」：问题改写 + 向量检索...")
            rewritten = rewrite_user_question(text)
            #print("解读的用户问题:", rewritten)
            seed_entities = query_prompt_desc_collection(rewritten, top_k=30)
            #print("检索的算子",seed_entities)
            if not seed_entities:
                raise ValueError("No candidate operators obtained.")

            # 调用新方法二（不再需要全图）
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
    # 3️⃣ 检查候选结果
    # ========================
    if len(candidates_nodes) == 0:
        return "No relevant operator information found; please try a more specific description."

    # print("\n🔍 最终候选节点:")
    for n in candidates_nodes:
        title = n.get('Title', n.get('Name', 'DataType'))
        source = n.get('__source__', 'Unknown')
        labels = ", ".join(n.get('Labels', [])) if isinstance(n.get('Labels'), list) else n.get('Labels', 'Unknown')
        # print(f"  - {title} (来源: {source}, 标签: {labels})")

    # ========================
    # 4️⃣ 构建 Prompt 并生成答案
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
            seen_generic = set()  # 普通关系去重: (rel_path, node_name)
            seen_datatype = set()  # hasDataType 去重: (param_name,)
            for item in reachable_info:
                rel_path = item["relation_path"]
                final_node = item["final_node"]
                path_nodes = item["path_nodes"]

                # 获取终点名称和类型
                n_title = (final_node.get('Title')or final_node.get('Name')or final_node.get('DataType'))
                n_labels = final_node.get('Labels', [])
                n_type = n_labels[0] if isinstance(n_labels, list) and n_labels else str(n_labels)

                # ✅ 特殊处理 hasDataType：显示它属于哪个参数
                if "hasDataType" in rel_path and len(path_nodes) >= 3:
                    # path_nodes[-2] 是参数节点（如 INPUT_RASTER）
                    param_node = path_nodes[-2]
                    param_name = param_node.get('Title', param_node.get('Name', 'UnknownParam'))
                    # 唯一键：参数名
                    if param_name in seen_datatype:
                        continue
                    seen_datatype.add(param_name)

                    info_parts.append(
                        f"  - Parameter '{param_name}' Data type: {n_title} ({n_type}) "
                        f"[{final_node.get('DataType', 'N/A')}]"
                    )
                else:
                    # 普通关系：按 (路径, 节点名) 去重
                    unique_key = (rel_path, n_title)
                    if unique_key in seen_generic:
                        continue
                    seen_generic.add(unique_key)

                    # 普通关系
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

    # 调用 LLM
    try:
        answer = call_turing_api(final_prompt)
        if answer.startswith("Error"):
            raise RuntimeError(f"LLM call failed: {answer}")
        return answer
    except Exception as e:
        return f"Error occurred while generating the answer: {str(e)}"


# 主程序
if __name__ == '__main__':
    try:
        text = input("Please enter the text content: ")
        result = process_text(text)
        print("result:", result)
    finally:
        weaviate_client.close()  # 确保关闭
        #print("✅ Weaviate connection has been closed.")
