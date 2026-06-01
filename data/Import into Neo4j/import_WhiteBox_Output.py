import pandas as pd
from py2neo import Graph, Node, Relationship

# Connect to Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

# Read CSV file (no header)
data = pd.read_csv('White_Output.csv',encoding="GBK", header=None)

# Iterate through each row of data
for index, row in data.iterrows():
    # Find Operator node using fixed index
    operator_id = row[0]
    operator_node = graph.nodes.match("Operation", ID=operator_id).first()
    if not operator_node:
        print(f"Operator with ID {operator_id} not found.")
        continue
    print(len(row))
    # Check if there is enough data to create GenericOutput node
    if len(row) < 3:

        print(f"Insufficient data in row {index} to create GenericOutput nodes.")
        continue

    # Starting from the 2nd column, build a GenericOutput node every 2 columns
    for i in range(1, len(row), 2):
        if i + 1 < len(row) and pd.notna(row[i]) and pd.notna(row[i + 1]):
            # Find or create GenericOutput node
            generic_output_node = graph.nodes.match("Output", Title=row[i], Description=row[i + 1]).first()
            if not generic_output_node:
                generic_output_node = Node("Output", Title=row[i], Description=row[i + 1])
                graph.create(generic_output_node)

            # Create Operator hasOutput relationship to GenericOutput
            has_output_relationship = graph.relationships.match((operator_node, generic_output_node),
                                                                        "hasOutput").first()
            if not has_output_relationship:
                has_output_relationship = Relationship(operator_node, "hasOutput", generic_output_node)
                graph.create(has_output_relationship)
        else:
            print(
                f"Insufficient data in row {index} to create a complete GenericOutput node from columns {i} and {i + 1}.")
