import pandas as pd
from py2neo import Graph, Node, Relationship

# Connect to Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

# Read CSV file (no header)
data = pd.read_csv('Catalyst_Output.csv', encoding="GBK", header=None)

# Iterate through each row of data
for index, row in data.iterrows():
    # Find Operator node using fixed index
    operator_id = row[0]
    operator_node = graph.nodes.match("Operation", ID=operator_id).first()
    if not operator_node:
        print(f"Operator with ID {operator_id} not found.")
        continue

    # Check if there is enough data to create GenericOutput node
    if len(row) < 3:
        print(f"Insufficient data in row {index} to create GenericOutput nodes.")
        continue

    # Starting from the 2nd column, build a group every 4 columns
    for i in range(1, len(row), 4):
        if i + 3 < len(row) and pd.notna(row[i]) and pd.notna(row[i + 1]) and pd.notna(row[i + 2]) and pd.notna(row[i + 3]):
            # Find or create GenericOutput node
            output_node = graph.nodes.match("Output", Title=row[i], Description=row[i + 1]).first()
            if not output_node:
                output_node = Node("Output", Title=row[i], Description=row[i + 1])
                graph.create(output_node)

            # Create Operator hasOutput relationship to GenericOutput
            has_output_relationship = graph.relationships.match((operator_node, output_node), "hasOutput").first()
            if not has_output_relationship:
                has_output_relationship = Relationship(operator_node, "hasOutput", output_node)
                graph.create(has_output_relationship)

            # Process the last two columns of data
            type_value = row[i + 3]
            data_type = row[i + 2]
            labels = ["ComplexData"]

            if type_value == "Vector":
                labels.extend(["Vector", "GeoData"])
            elif type_value == "Raster":
                labels.extend(["Raster", "GeoData"])
            elif type_value == "Geodata":
                labels.append("GeoData")
            elif type_value == "NonGeoData":
                labels.append("NonGeoData")
            elif type_value == "LiteralData":
                labels = ["LiteralData"]

            # Create node with dynamic labels
            node_labels = ":".join(labels)
            query = f"""
            MERGE (n:{node_labels} {{DataType: $data_type}})
            RETURN n
            """
            result = graph.run(query, data_type=data_type).data()
            special_node = result[0]['n']

            # Create Operator hasSpecialNode SpecialNode relationship
            has_special_node_relationship = graph.relationships.match((output_node, special_node), "hasDataType").first()
            if not has_special_node_relationship:
                has_special_node_relationship = Relationship(output_node, "hasDataType", special_node)
                graph.create(has_special_node_relationship)



