import json
import csv
import os

# Process a single JSON file and extract all Output section fields
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize row list
    row = []

    # Process Output section
    outputs = data.get('Output', [])

    # If Output exists, append all Output fields to the row list
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
        # If no Output data, append empty values to the row list
        row.extend([''] * 5)  # 5 is based on the number of fields per Output row

    return row

# Directory containing JSON files
input_dir = 'json_datas'  # Update this to the directory containing JSON files

# Output CSV file
output_csv = 'Output.csv'

# Open CSV file for writing
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Process all JSON files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            row = process_json_file(file_path)
            writer.writerow(row)

print(f"Data has been written: {output_csv}")
