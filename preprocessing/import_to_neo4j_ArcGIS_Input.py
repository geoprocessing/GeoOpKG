import pandas as pd
from py2neo import Graph, Node, Relationship

# 请设置连接到Neo4j数据库的信息
graph = Graph("bolt://localhost:7687", auth=("", ""))

# 读取CSV文件（无表头）
data = pd.read_csv('ArcGIS_Input.csv', encoding="GBK", header=None)

# 遍历每行数据
for index, row in data.iterrows():
    # 使用固定索引查找或创建Software节点
    software_node = graph.nodes.match("Software", Name=row[0], version=row[1]).first()
    if not software_node:
        software_node = Node("Software", Name=row[0], version=row[1])
        graph.create(software_node)

    # 使用固定索引查找或创建Operator节点
    operator_node = graph.nodes.match("Operation", Title=row[2], ID=row[3], Description=row[4]).first()
    if not operator_node:
        operator_node = Node("Operation", Title=row[2], ID=row[3], Description=row[4])
        graph.create(operator_node)

    # 创建Operator ImplementIn Software关系
    implement_relationship = graph.relationships.match((operator_node, software_node), "ImplementedIn").first()
    if not implement_relationship:
        implement_relationship = Relationship(operator_node, "ImplementedIn", software_node)
        graph.create(implement_relationship)

    # 从第六列开始，每四列构建一个组合
    for i in range(5, len(row), 4):
        if pd.notna(row[i]) and pd.notna(row[i + 1]):
            # 查找或创建GenericInput节点
            input_node = graph.nodes.match("Input", Title=row[i], Description=row[i + 1]).first()
            if not input_node:
                input_node = Node("Input", Title=row[i], Description=row[i + 1])
                graph.create(input_node)

            # 创建Operator hasGenericInput GenericInput关系
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

            # 创建数据类型的节点
            node_labels = ":".join(labels)
            query = f"""
            MERGE (n:{node_labels} {{DataType: $data_type}})
            RETURN n
            """
            result = graph.run(query, data_type=data_type).data()
            data_node = result[0]['n']

            # 创建Operator hasSpecialNode SpecialNode关系
            has_special_node_relationship = graph.relationships.match((input_node, data_node),
                                                                      "hasDataType").first()
            if not has_special_node_relationship:
                has_special_node_relationship = Relationship(input_node, "hasDataType", data_node)
                graph.create(has_special_node_relationship)

