from neo4j import GraphDatabase
import pandas as pd

# 请设置连接Neo4j数据库信息
uri = "bolt://localhost:7687"
username = ""
password = ""
driver = GraphDatabase.driver(uri, auth=(username, password))

def create_hierarchy(tx, row):
    # 提取所有列（假设列名为 A, B, C, D, E, F, G, H, I, J）
    A, B, C = row[0], row[1], row[2]
    D, E, F = row[3], row[4], row[5]
    G, H, I, J = row[6], row[7], row[8], row[9]

    # 存储有效的 Algorithm 节点信息 (name, description)
    algos = []
    if pd.notna(A) and A.strip():
        algos.append(('A', A.strip(), D.strip() if pd.notna(D) else ""))
    if pd.notna(B) and B.strip():
        algos.append(('B', B.strip(), E.strip() if pd.notna(E) else ""))
    if pd.notna(C) and C.strip():
        algos.append(('C', C.strip(), F.strip() if pd.notna(F) else ""))

    # 创建 Algorithm 节点（去重）
    for tag, name, desc in algos:
        tx.run("""
            MERGE (a:Algorithm {Name: $name})
            ON CREATE SET a.Description = $desc
            ON MATCH SET a.Description = CASE 
                WHEN a.Description IS NULL OR a.Description = '' THEN $desc 
                ELSE a.Description 
            END
        """, name=name, desc=desc)

    # 建立 PartOf 层级关系：B -> A, C -> B
    if pd.notna(B) and B.strip() and pd.notna(A) and A.strip():
        tx.run("""
            MERGE (child:Algorithm {Name: $child}) 
            MERGE (parent:Algorithm {Name: $parent})
            MERGE (child)-[:PartOf]->(parent)
        """, child=B.strip(), parent=A.strip())

    if pd.notna(C) and C.strip() and pd.notna(B) and B.strip():
        tx.run("""
            MERGE (child:Algorithm {Name: $child})
            MERGE (parent:Algorithm {Name: $parent})
            MERGE (child)-[:PartOf]->(parent)
        """, child=C.strip(), parent=B.strip())

    # 处理输入输出：仅当对应 Algorithm 存在时
    inputs_outputs = []

    # 子类1 的输入输出（绑定到 B）
    if pd.notna(B) and B.strip():
        if pd.notna(G) and G.strip():
            inputs_outputs.append(('input', G.strip(), B.strip()))
        if pd.notna(H) and H.strip():
            inputs_outputs.append(('output', H.strip(), B.strip()))

    # 子类2 的输入输出（绑定到 C）
    if pd.notna(C) and C.strip():
        if pd.notna(I) and I.strip():
            inputs_outputs.append(('input', I.strip(), C.strip()))
        if pd.notna(J) and J.strip():
            inputs_outputs.append(('output', J.strip(), C.strip()))

    for io_type, title, alg_name in inputs_outputs:
        if io_type == 'input':
            tx.run("""
                MERGE (gi:GenericInput {Title: $title})
                WITH gi
                MATCH (a:Algorithm {Name: $alg_name})
                MERGE (a)-[r:has_GenericInput]->(gi)
            """, title=title, alg_name=alg_name)
        else:  # output
            tx.run("""
                MERGE (go:GenericOutput {Title: $title})
                WITH go
                MATCH (a:Algorithm {Name: $alg_name})
                MERGE (a)-[r:has_GenericOutput]->(go)
            """, title=title, alg_name=alg_name)

# 读取数据
df = pd.read_csv("Algorithm.csv", encoding='gbk')  # 或 Excel
df = df.fillna('')  # 转换 NaN 为 ''

# 执行写入
with driver.session() as session:
    for _, row in df.iterrows():
        session.execute_write(create_hierarchy, row)

driver.close()