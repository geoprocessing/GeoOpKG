import pandas as pd
from py2neo import Graph, Node, Relationship

# Connect to Neo4j database
graph = Graph('bolt://localhost:7687', auth=('neo4j', ''))

# Read CSV file, without header
df = pd.read_csv("ArcGIS_Algorithm.csv", header=None, encoding="utf-8")

# Iterate through each row of DataFrame to create relationships
for index, row in df.iterrows():
    operator_id = str(row[0])  # Assume Operator ID is in the first column
    algorithm_name = row[1]    # Assume Algorithm name is in the second column

    # Find Operator node
    operator_query = f"MATCH (o:Operation) WHERE o.ID = {operator_id} RETURN o"

    operator_node = graph.evaluate(operator_query)
    if operator_node is None:
        print(f"No matching Operation node found for ID: {operator_id}")

    # Find Algorithm node
    algorithm_query = f"MATCH (a:Algorithm) WHERE a.Name = '{algorithm_name}' RETURN a"
    algorithm_node = graph.evaluate(algorithm_query)
    if algorithm_node is None:
        print(f"No matching Algorithm node found for Name: {algorithm_name}")

    if operator_node and algorithm_node:
        # Create relationship between Operator node and Algorithm node
        has_plan_relationship = Relationship(operator_node, "hasPlan", algorithm_node)
        graph.create(has_plan_relationship)
    else:
        print(f"No matching nodes found for Operator ID: {operator_id} or Algorithm Name: {algorithm_name}")

print("Process completed.")
