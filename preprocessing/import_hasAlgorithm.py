import pandas as pd
from py2neo import Graph, Node, Relationship

# 连接到 Neo4j 数据库
graph = Graph('bolt://localhost:7687', auth=('neo4j', 'Yq123456'))

# 读取 CSV 文件，没有表头
df = pd.read_csv("ArcGIS_Algorithm.csv", header=None, encoding="utf-8")

# 遍历 DataFrame 的每一行，创建关系
for index, row in df.iterrows():
    operator_id = str(row[0])  # 假设算子ID在第一列
    algorithm_name = row[1]    # 假设算法名称在第二列

    # 查找 Operator 节点
    operator_query = f"MATCH (o:Operation) WHERE o.ID = {operator_id} RETURN o"

    operator_node = graph.evaluate(operator_query)
    if operator_node is None:
        print(f"No matching Operation node found for ID: {operator_id}")

    # 查找 Algorithm 节点
    algorithm_query = f"MATCH (a:Algorithm) WHERE a.Name = '{algorithm_name}' RETURN a"
    algorithm_node = graph.evaluate(algorithm_query)
    if algorithm_node is None:
        print(f"No matching Algorithm node found for Name: {algorithm_name}")

    if operator_node and algorithm_node:
        # 创建 Operator 节点和 Algorithm 节点之间的关系
        has_plan_relationship = Relationship(operator_node, "hasPlan", algorithm_node)
        graph.create(has_plan_relationship)
    else:
        print(f"No matching nodes found for Operator ID: {operator_id} or Algorithm Name: {algorithm_name}")

print("Process completed.")
