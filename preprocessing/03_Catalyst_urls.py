from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import pandas as pd

url = "https://catalyst.earth/catalyst-system-files/professional-help/references/algoreference_r/modeler/M_u.html"

with sync_playwright() as p:
    # 使用 Chromium 浏览器
    browser = p.chromium.launch(headless=True)  # headless=True 静默运行
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    page = context.new_page()
    page.goto(url)

    # 等待页面加载完成
    page.wait_for_load_state("networkidle")

    # 获取所有相关的 <a> 标签
    links = page.query_selector_all("div.related-links a")

    urls = []
    for link in links:
        href = link.get_attribute("href")
        if href and "pciFunction_r" in href:  # 只要 pciFunction_r 的链接
            full_url = urljoin(url, href)
            urls.append(full_url)

    browser.close()

# 保存到 CSV，没有表头
df = pd.DataFrame(urls)
df.to_csv("U_urls.csv", index=False, header=False, encoding="utf-8")

print("Data has been written: U_urls.csv")
