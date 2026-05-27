from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import pandas as pd

url = "https://catalyst.earth/catalyst-system-files/professional-help/references/algoreference_r/modeler/M_u.html"

with sync_playwright() as p:
    # Use Chromium browser
    browser = p.chromium.launch(headless=True)  # headless=True run silently
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    page = context.new_page()
    page.goto(url)

    # Wait for page to finish loading
    page.wait_for_load_state("networkidle")

    # Get all related <a> tags
    links = page.query_selector_all("div.related-links a")

    urls = []
    for link in links:
        href = link.get_attribute("href")
        if href and "pciFunction_r" in href:  # Only links containing pciFunction_r
            full_url = urljoin(url, href)
            urls.append(full_url)

    browser.close()

# Save to CSV without header
df = pd.DataFrame(urls)
df.to_csv("U_urls.csv", index=False, header=False, encoding="utf-8")

print("Data has been written: U_urls.csv")
