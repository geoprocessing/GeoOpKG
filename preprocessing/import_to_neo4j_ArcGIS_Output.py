import pandas as pd
from py2neo import Graph, Node, Relationship

# 请设置连接到Neo4j数据库的信息
graph = Graph("bolt://localhost:7687", auth=("", ""))

# 读取CSV文件（无表头）
data = pd.read_csv('ArcGIS_Output.csv', encoding="GBK", header=None)

# 遍历每行数据
for index, row in data.iterrows():
    # 使用固定索引查找Operator节点
    operator_id = row[0]
    operator_node = graph.nodes.match("Operation", ID=operator_id).first()
    if not operator_node:
        print(f"Operator with ID {operator_id} not found.")
        continue

    # 检查是否有足够的数据来创建GenericOutput节点
    if len(row) < 3:
        print(f"Insufficient data in row {index} to create GenericOutput nodes.")
        continue

    # 从第二列开始，每四列构建一个组合
    for i in range(1, len(row), 4):
        if i + 3 < len(row) and pd.notna(row[i]) and pd.notna(row[i + 1]) and pd.notna(row[i + 2]) and pd.notna(row[i + 3]):
            # 查找或创建GenericOutput节点
            output_node = graph.nodes.match("Output", Title=row[i], Description=row[i + 1]).first()
            if not output_node:
                output_node = Node("Output", Title=row[i], Description=row[i + 1])
                graph.create(output_node)

            # 创建Operator hasGenericOutput GenericOutput关系
            has_output_relationship = graph.relationships.match((operator_node, output_node), "hasOutput").first()
            if not has_output_relationship:
                has_output_relationship = Relationship(operator_node, "hasOutput", output_node)
                graph.create(has_output_relationship)

            # 处理后两列数据
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

            # 创建具有动态标签的节点
            node_labels = ":".join(labels)
            query = f"""
            MERGE (n:{node_labels} {{DataType: $data_type}})
            RETURN n
            """
            result = graph.run(query, data_type=data_type).data()
            special_node = result[0]['n']

            # 创建Operator hasSpecialNode SpecialNode关系
            has_special_node_relationship = graph.relationships.match((output_node, special_node), "hasDataType").first()
            if not has_special_node_relationship:
                has_special_node_relationship = Relationship(output_node, "hasDataType", special_node)
                graph.create(has_special_node_relationship)



