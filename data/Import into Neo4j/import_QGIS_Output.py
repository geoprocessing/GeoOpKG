import pandas as pd
from py2neo import Graph, Node, Relationship

# Connect to Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

# Read CSV file (no header)
data = pd.read_csv('QGIS_Output.csv', encoding="GBK", header=None)

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
    for i in range(1, len(row), 5):
        if i + 3 < len(row) and pd.notna(row[i]) and pd.notna(row[i + 1]) and pd.notna(row[i + 2]) and pd.notna(row[i + 3]):
            # Find or create GenericOutput node
            output_node = graph.nodes.match("Output", Title=row[i],Description=row[i+1], Identifier=row[i + 2]).first()
            if not output_node:
                output_node = Node("Output", Title=row[i],Description=row[i+1], Identifier=row[i + 2])
                graph.create(output_node)

            # Create Operator hasOutput relationship to GenericOutput
            has_output_relationship = graph.relationships.match((operator_node, output_node), "hasOutput").first()
            if not has_output_relationship:
                has_output_relationship = Relationship(operator_node, "hasOutput", output_node)
                graph.create(has_output_relationship)

            # Process the last two columns of data
            data_type = row[i + 3]
            data_format = row[i + 4]
            labels = ["ComplexData"]

            if data_type == "vector":
                labels.extend(["Vector", "GeoData"])
            elif data_type == "raster":
                labels.extend(["Raster", "GeoData"])
            elif data_type == "geographicData":
                labels.append("GeoData")
            elif data_type == "NonGeoData":
                labels.append("NonGeoData")
            elif data_type == "LiteralData":
                labels = ["LiteralData"]

            # Create data type node
            node_labels = ":".join(labels)
            if node_labels == 'LiteralData':
                query = f"""
                        MERGE (n:{node_labels} {{DataType: $data_format}})
                        RETURN n
                        """
                result = graph.run(query, data_format=data_format).data()
                data_node = result[0]['n']
            else:
                query = f"""
                        MERGE (n:{node_labels} {{DataType: $data_type,DataFormat: $data_format}})
                        RETURN n
                        """
                result = graph.run(query, data_type=data_type, data_format=data_format).data()
                data_node = result[0]['n']

            # Create Operator hasSpecialNode SpecialNode relationship
            has_data_node_relationship = graph.relationships.match((output_node, data_node), "hasDataType").first()
            if not has_data_node_relationship:
                has_data_node_relationship = Relationship(output_node, "hasDataType", data_node)
                graph.create(has_data_node_relationship)



