import json
import csv
import os

# 处理单个JSON文件并提取必要信息的函数
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data.get('Title', '')
    abstract = data.get('Abstract', '')
    identifier = data.get('Identifier', '')

    row = [title, abstract, identifier]

    inputs = data.get('Input', [])
    for input_item in inputs:
        input_title = input_item.get('Title', '')
        input_abstract = input_item.get('Abstract', '')
        input_identifier = input_item.get('Identifier', '')
        min_occurs = input_item.get('minOccurs', '')
        max_occurs = input_item.get('maxOccurs', '')

        data_type = input_item.get('DataType', '')

        if data_type == 'ComplexData':
            complex_data = input_item.get('ComplexData', {})
            complex_data_type = complex_data.get('data_type', '')
            formats = ','.join([fmt.get('mimeType', '') for fmt in complex_data.get('Format', [])])
            row.extend([input_title, input_abstract, input_identifier, min_occurs, max_occurs, complex_data_type, formats])
        elif data_type == 'LiteralData':
            literal_data = input_item.get('LiteralData', {})
            literal_data_domain = literal_data.get('LiteralDataDomain', [{}])[0]
            literal_data_type = 'LiteralData'
            literal_data_content = literal_data_domain.get('DataType', {}).get('content', '')
            row.extend([input_title, input_abstract, input_identifier, min_occurs, max_occurs, literal_data_type, literal_data_content])
        else:
            row.extend([input_title, input_abstract, input_identifier, min_occurs, max_occurs, '', ''])

    return row

# JSON文件所在目录
input_dir = 'json_datas'

# 输出CSV文件
input_csv = 'Input.csv'

rows = []

# 处理目录中的所有JSON文件
for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(input_dir, file_name)
        row = process_json_file(file_path)
        rows.append(row)

# 找到最长的行，用于对齐CSV列
max_len = max(len(row) for row in rows)

# **🔹 重点修改部分**
def format_cell(value):
    """如果值以 '-' 开头，就加上双引号"""
    if isinstance(value, str) and value.startswith('-'):
        return f'\t{value}'
    return value

# **写入 CSV**
with open(input_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)  # 只加必要的引号
    for row in rows:
        # **对行中每个单元格进行检查**
        formatted_row = [format_cell(cell) for cell in row]

        # 补全较短的行
        if len(formatted_row) < max_len:
            formatted_row.extend([''] * (max_len - len(formatted_row)))

        writer.writerow(formatted_row)

print(f"Data has been written: {input_csv}")
