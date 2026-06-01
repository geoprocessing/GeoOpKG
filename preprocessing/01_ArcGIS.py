import requests
from bs4 import BeautifulSoup as bs, BeautifulSoup
from openpyxl import Workbook
from fake_useragent import UserAgent
import pandas as pd

# Toolbox URL - toolset - tools

# Crawl toolset URLs from the toolbox
url = "https://desktop.arcgis.com/en/arcmap/10.3/tools/workflow-manager-toolbox/overview-of-wmx-tools.htm"
# Request header
header = {'User-Agent': UserAgent().random}
response = requests.get(url, headers=header)

# Get page content
soup = BeautifulSoup(response.content, 'html.parser')
# Parse table data
tr_tags = soup.find_all('tbody', class_="align-middle")
# Toolset URL list
Toolset_list = []
# Locate href
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

# Traverse toolsets
for j in range(0, len(Toolset_list)):
    url = "https://desktop.arcgis.com" + Toolset_list[j]
    header = {'User-Agent': UserAgent().random}
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Parse table data
    tr_tags = soup.find_all('tbody', class_="align-middle")

    # Tool URL list
    Tool_list = []
    In = []
    Op = []
    # Locate href
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
    # Traverse tools
    for r in range(0, len(Tool_list)):
        url = "https://desktop.arcgis.com" + Tool_list[r]

        header = {'User-Agent': UserAgent().random}
        response = requests.get(url, headers=header)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Parse table data
        tr_tags = soup.find('table', class_="gptoolparamtbl")
        list = []
        #print(len(tr_tags))
        # Iterate through tr tags and extract td text
        td_tags = tr_tags.select('tr td')
        # print(td_tags)
        for i in range(3, len(td_tags)):
            Data = td_tags[i].get_text()
            # print(Data)
            list.append(Data)
        In.append(list)
    Operator.append(Op)
    Input.append(In)

df = pd.DataFrame(Operator)  # Create a DataFrame
df.to_excel(r"Op.xlsx", index=False)  # Export DataFrame to Excel

df = pd.DataFrame(Input)  # Create a DataFrame
df.to_excel(r"In.xlsx", index=False)  # Export DataFrame to Excel
