import json
import os
import time
import pandas as pd
from http import HTTPStatus
import dashscope
from dashscope import Generation


# 请设置API-KEY
dashscope.api_key = ""
# 定义问题和消息列表模板
messages_template = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': ''}
]

# 读取算子数据和类别数据
operators_df = pd.read_csv('ArcGIS_Input.csv', encoding='latin1', header=None)

# 读取类别CSV，假设列顺序为：
# Level 1;Level 2;Level 3;Level 1 Description;Level 2 Description;Level 3 Description;Level 2 InputType;Level 2 OutputType;Level 3 InputType;Level 3 OutputType
categories_df = pd.read_csv('Algorithm.csv', encoding='GBK', header=None)

# 创建类别字典
# 结构: { Level1: { 'description': ..., 'subcategories': { Level2: { 'description': ..., 'InputType': ..., 'OutputType': ..., 'subcategories': { Level3: { 'description': ..., 'InputType': ..., 'OutputType': ... } } } } } }
categories_dict = {}

for _, row in categories_df.iterrows():
    level1 = row[0]
    level2 = row[1] if pd.notna(row[1]) else None
    level3 = row[2] if pd.notna(row[2]) else None
    level1_desc = row[3] if pd.notna(row[3]) else ''
    level2_desc = row[4] if pd.notna(row[4]) else ''
    level3_desc = row[5] if pd.notna(row[5]) else ''
    level2_in = row[6] if pd.notna(row[6]) else ''
    level2_out = row[7] if pd.notna(row[7]) else ''
    level3_in = row[8] if pd.notna(row[8]) else ''
    level3_out = row[9] if pd.notna(row[9]) else ''

    if level1 not in categories_dict:
        categories_dict[level1] = {'description': level1_desc, 'subcategories': {}}

    if level2:
        if level2 not in categories_dict[level1]['subcategories']:
            categories_dict[level1]['subcategories'][level2] = {
                'description': level2_desc,
                'InputType': level2_in,
                'OutputType': level2_out,
                'subcategories': {}
            }

        if level3:
            categories_dict[level1]['subcategories'][level2]['subcategories'][level3] = {
                'description': level3_desc,
                'InputType': level3_in,
                'OutputType': level3_out
            }

# 函数：检查分类结果是否有效（支持 Level3/Level2/Level1）
def is_valid_classification(classification_text):
    for level1, info1 in categories_dict.items():
        if classification_text == level1:
            return True
        for level2, info2 in info1['subcategories'].items():
            if classification_text == level2:
                return True
            for level3 in info2['subcategories'].keys():
                if classification_text == level3:
                    return True
    return False

# 对单个算子调用模型进行分类·
def classify_operator(software, name, description, parameters):
    prompt = (
        f"You are a GIS domain expert.\n"
        f"Your task is to classify GIS operations strictly into one of the following categories.\n"
        f"The categories are hierarchical: Level 1 > Level 2 > Level 3.\n"
        f"Level 1 is the most abstract and general."
        f"Level 2 is more specific than Level 1."
        f"Level 3 is the most detailed and concrete."
        f"Classification must be based on the category name and its description, as well as your own understanding.\n"
        f"Always select the lowest available level when classifying (if it exists, choose level 3)" 
        f"Only fall back to Level 2 or Level 1 if absolutely no Level 3 fits.\n "
        "Important rules:\n"
        "1.Always classify into the lowest available level (prefer Level 3 if available; otherwise Level 2; otherwise Level 1)."
        "2.You must output only one category, not a full path.\n"
        "3.The output format must be strictly: Category - Explanation, with exactly one line.\n"
        "Example of correct output: Format conversion - Convert datasets between different formats\n"
        "Incorrect: {Raster spatial analysis and calculation → Statistical analysis → Neighborhood statistics - ...}"
        "4.Do not invent new categories or modify category names.\n"
        "5.The classification must be based on the information of the operation and categorized into the most suitable category.\n\n"
    )
    prompt += "\nClassification options:\n"
    for level1, info1 in categories_dict.items():
        prompt += f"{level1} - {info1['description']}\n"
        for level2, info2 in info1['subcategories'].items():
            prompt += f"  {level2} - {info2['description']}\n"
            for level3, info3 in info2['subcategories'].items():
                prompt += f"    {level3} - {info3['description']}\n"

    prompt += (
        f"\nNow classify the following GIS operation:\n"
        f"---\n"
        f"Software: {software}"
        f"Operator Name: {name}\n"
        f"Description: {description}\n"
        f"Parameters:\n")

    for param_name, param_desc,param_extra in parameters:
        prompt += f"  {param_name}:{param_extra} {param_desc}\n"

    messages = messages_template.copy()
    messages[1]['content'] = prompt

    try:
        response = Generation.call(
            model='qwen3-max-2025-09-23',
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            result = response.output['choices'][0]['message']['content'].strip()
            print(result)
            # 提取类别名称
            category = result.split(' - ')[0] if " - " in result else result
            print(category)
            return category
        else:
            print(f"Request id: {response.request_id}, Status code: {response.status_code}")
            return "Classification failed."
    except Exception as e:
        print(f"Error in API call: {e}")
        return "Classification failed."


# 函数：对单个算子进行分类（带重试机制）
def classify_operator_with_retry(software, name, description, parameters, max_retries=1):
    retries = 0
    classification_text = None

    while retries < max_retries:
        classification_text = classify_operator(software, name, description, parameters)
        if is_valid_classification(classification_text):
            return classification_text
        else:
            #print(f"分类不符合 类别 - 解释 规范，重试第 {retries + 1} 次...")
            #print(f"classification_text:",classification_text)
            retries += 1
            time.sleep(1)  # 加延迟，避免频繁调用API

    #print("多次尝试后分类仍无效，跳过...")
    return "Classification failed."

# 判断分类是否最低级
def is_lowest_level(category_name):
    for level1, info1 in categories_dict.items():
        if category_name == level1:
            # Level1，如果有子类，则不是最低级
            return len(info1['subcategories']) == 0
        for level2, info2 in info1['subcategories'].items():
            if category_name == level2:
                # Level2，如果有子类，则不是最低级
                return len(info2['subcategories']) == 0
            for level3 in info2['subcategories'].keys():
                if category_name == level3:
                    # Level3没有子类，肯定最低级
                    return True
    return True  # 如果找不到，默认认为最低级

# 为每个算子添加分类
batch_size = 7  # 每次处理5个描述
classified_operators = []

for i in range(0, len(operators_df), batch_size):
    batch = operators_df.iloc[i:i + batch_size]
    for j, (_, row) in enumerate(batch.iterrows()):
        software = row[0]  # 算子所在软件
        name = row[2]  # 算子名称
        operator_id = row[3]  # 算子ID
        description = row[4]  # 算子描述

        # 处理参数信息
        parameters = []
        num_cols_per_param = 4
        for k in range(5, len(row),4):
            param_name = row[k]#参数第一列
            param_desc = row[k + 1] if k + 1 < len(row) else ''#参数第二列
            param_extra = row[k + 3] if k + 3 < len(row) else ''  # 第4列

            if pd.notna(param_name):
                parameters.append((param_name, param_desc, param_extra))

        # 分类（带补偿机制）
        classification_text = classify_operator_with_retry(software, name, description, parameters)

        # 判断是否最低级
        lowest_flag = 0 if is_lowest_level(classification_text) else 1

        classified_operators.append([operator_id, classification_text,lowest_flag])

        # 避免频繁API调用，加上延迟
        time.sleep(1)

# 保存分类结果
if classified_operators:
    classified_df = pd.DataFrame(classified_operators, columns=['Operator ID', 'Classification', 'NotLowest'])
    classified_df.to_csv('ArcGIS_Algorithm.csv', index=False, encoding='gbk')
    #print("分类结果已保存!")
else:
    print("No classification results to save.")
