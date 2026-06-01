import pandas as pd
from py2neo import Graph, Node, Relationship

# Connect to Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

# Read CSV file (no header)
data = pd.read_csv('ArcGIS_Input.csv', encoding="GBK", header=None)

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

    # Starting from the 6th column, build a group every 4 columns
    for i in range(5, len(row), 4):
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

        if pd.notna(row[i + 2]) and pd.notna(row[i + 3]):
            type_value = row[i + 2]
            data_type = row[i + 3]
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

            # Create data type node
            node_labels = ":".join(labels)
            query = f"""
            MERGE (n:{node_labels} {{DataType: $data_type}})
            RETURN n
            """
            result = graph.run(query, data_type=data_type).data()
            data_node = result[0]['n']

            # Create Operator hasSpecialNode SpecialNode relationship
            has_special_node_relationship = graph.relationships.match((input_node, data_node),
                                                                      "hasDataType").first()
            if not has_special_node_relationship:
                has_special_node_relationship = Relationship(input_node, "hasDataType", data_node)
                graph.create(has_special_node_relationship)

