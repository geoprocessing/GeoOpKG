import pandas as pd
from py2neo import Graph, Node, Relationship

# Connect to Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

# Read CSV file (no header)
data = pd.read_csv('White_Input.csv',encoding="GBK", header=None)

# Iterate through each row of data
for index, row in data.iterrows():
    # Find or create Software node using fixed index
    software_node = graph.nodes.match("Software", Name=row[0], version=row[1]).first()
    if not software_node:
        software_node = Node("Software", Name=row[0], version=row[1])
        graph.create(software_node)

    # Find or create Operator node using fixed index
    operator_node = graph.nodes.match("Operation", Title=row[2], ID=row[3], Description=row[4]).first()
    if not operator_node:
        operator_node = Node("Operation", Title=row[2], ID=row[3], Description=row[4])
        graph.create(operator_node)

    # Create Operator ImplementedIn Software relationship
    implement_relationship = graph.relationships.match((operator_node, software_node), "ImplementedIn").first()
    if not implement_relationship:
        implement_relationship = Relationship(operator_node, "ImplementedIn", software_node)
        graph.create(implement_relationship)

    # Starting from the 6th column, build a GenericInput node every 2 columns
    for i in range(5, len(row), 2):
        if pd.notna(row[i]) and pd.notna(row[i + 1]):
            # Find or create GenericInput node
            input_node = graph.nodes.match("Input", Title=row[i], Description=row[i + 1]).first()
            if not input_node:
                input_node = Node("Input", Title=row[i], Description=row[i + 1])
                graph.create(input_node)

            # Create Operator hasInput relationship to GenericInput
            has_input_relationship = graph.relationships.match((operator_node, input_node),
                                                                       "hasInput").first()
            if not has_input_relationship:
                has_input_relationship = Relationship(operator_node, "hasInput", input_node)
                graph.create(has_input_relationship)
