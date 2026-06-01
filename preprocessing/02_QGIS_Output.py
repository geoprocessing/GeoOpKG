import json
import csv
import os

# 处理单个 JSON 文件并提取所有 Output 部分信息的函数
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化行列表
    row = []

    # 处理 Output 部分
    outputs = data.get('Output', [])

    # 如果有 Output 数据，将所有 Output 的信息合并到行列表中
    if outputs:
        for output_item in outputs:
            output_title = output_item.get('Title', '')
            output_abstract = output_item.get('Abstract', '')
            output_identifier = output_item.get('Identifier', '')

            data_type = output_item.get('DataType', '')

            if data_type == 'ComplexData':
                complex_data = output_item.get('ComplexData', {})
                complex_data_type = complex_data.get('data_type', '')
                formats = ','.join([fmt.get('mimeType', '') for fmt in complex_data.get('Format', [])])
                row.extend([str(output_title), str(output_abstract), str(output_identifier), str(complex_data_type), str(formats)])
            elif data_type == 'LiteralData':
                literal_data = output_item.get('LiteralData', {})
                literal_data_domain = literal_data.get('LiteralDataDomain', [{}])[0]
                literal_data_type = 'LiteralData'
                literal_data_content = literal_data_domain.get('DataType', {}).get('content', '')
                row.extend([str(output_title), str(output_abstract), str(output_identifier), str(literal_data_type), str(literal_data_content)])
            else:
                row.extend([str(output_title), str(output_abstract), str(output_identifier), '', ''])
    else:
        # 如果没有 Output 数据，添加空值到行列表中
        row.extend([''] * 5)  # 5 是根据 Output 的每行数据中的内容计算得出

    return row

# JSON 文件所在目录
input_dir = 'json_datas'  # 将此处修改为包含 JSON 文件的目录

# 输出 CSV 文件
output_csv = 'Output.csv'

# 打开 CSV 文件准备写入
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # 处理目录中的所有 JSON 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            row = process_json_file(file_path)
            writer.writerow(row)

print(f"Data has been written: {output_csv}")
