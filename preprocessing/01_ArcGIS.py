import requests
from bs4 import BeautifulSoup as bs, BeautifulSoup
from openpyxl import Workbook
from fake_useragent import UserAgent
import pandas as pd

#工具箱url-工具集-工具

# 爬取工具箱中工具集的地址
url = "https://desktop.arcgis.com/en/arcmap/10.3/tools/workflow-manager-toolbox/overview-of-wmx-tools.htm"
# 请求header
header = {'User-Agent': UserAgent().random}
response = requests.get(url, headers=header)

# 获取网页信息
soup = BeautifulSoup(response.content, 'html.parser')
# 到表格数据
tr_tags = soup.find_all('tbody', class_="align-middle")
# 工具集URL列表
Toolset_list = []
# 定位到href
for tr_tags in tr_tags:
    td_tags = tr_tags.select('tr td')
    if len(td_tags)<12:
        continue
    for i in range(0, len(td_tags),2):
        a_tags = td_tags[i].select('a')
        #print(a_tags)
        toolset_href = a_tags[0].get('href')

        Toolset_list.append(toolset_href)
    print(Toolset_list)

Operator = []
Input = []

# 遍历工具集
for j in range(0, len(Toolset_list)):
    url = "https://desktop.arcgis.com" + Toolset_list[j]
    header = {'User-Agent': UserAgent().random}
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.content, 'html.parser')
    # 到表格数据
    tr_tags = soup.find_all('tbody', class_="align-middle")

    # 工具url
    Tool_list = []
    In = []
    Op = []
    # 定位到href
    for tr_tags in tr_tags:
        td_tags = tr_tags.select('tr td')
        #print(td_tags)
        if len(td_tags) % 2 != 0:
            continue
        for i in range(0, len(td_tags), 2):
            a_tags = td_tags[i].select('a')
            #print(a_tags)
            if a_tags==[]:
                continue
            tool_href = a_tags[0].get('href')
            Tool_list.append(tool_href)
        list1 = []
        for i in range(0, len(td_tags), 2):
            Title = td_tags[i].get_text()
            list1.append(Title)
            Abstract = td_tags[i + 1].get_text()
            list1.append(Abstract)
        Op.append(list1)
    #print(Tool_list)
    # 遍历工具
    for r in range(0, len(Tool_list)):
        url = "https://desktop.arcgis.com" + Tool_list[r]

        header = {'User-Agent': UserAgent().random}
        response = requests.get(url, headers=header)
        soup = BeautifulSoup(response.content, 'html.parser')
        # 到表格数据
        tr_tags = soup.find('table', class_="gptoolparamtbl")
        list = []
        #print(len(tr_tags))
        # 循环遍历获取tr标签下的td标签文本
        td_tags = tr_tags.select('tr td')
        # print(td_tags)
        for i in range(3, len(td_tags)):
            Data = td_tags[i].get_text()
            # print(Data)
            list.append(Data)
        In.append(list)
    Operator.append(Op)
    Input.append(In)

df = pd.DataFrame(Operator)  # 创建一个 DataFrame
df.to_excel(r"Op.xlsx", index=False)  # 将 DataFrame 导出为 Excel 文件

df = pd.DataFrame(Input)  # 创建一个 DataFrame
df.to_excel(r"In.xlsx", index=False)  # 将 DataFrame 导出为 Excel 文件
