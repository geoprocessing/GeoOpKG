from playwright.sync_api import sync_playwright
import csv
import re

# 输入和输出文件
INPUT_URLS_FILE = "urls.csv"
OUTPUT_CSV_FILE = "Input.csv"

def clean_name(text: str) -> str:
    """清理参数名，去掉末尾的 * 和空格"""
    return re.sub(r"\s*\*$", "", text.strip())


def extract_algorithm_data(page, url):
    page.goto(url)
    page.wait_for_load_state("networkidle", timeout=10000)
    # 算法名称
    try:
        algo_name = page.locator("h1.topictitle1").inner_text().strip()
    except:
        algo_name = "Unknown"
    # 算法描述
    try:
        # Description 块（适配 ports 页面和 parameters 页面）
        desc_block = ""
        if page.locator("div[id$='_Description']").count() > 0:
            desc_block = page.locator("div[id$='_Description']").inner_text().strip()
        elif page.locator("div.section[id*='__sec_description']").count() > 0:
            paras = page.locator("div.section[id*='__sec_description'] p.p")
            desc_parts = [paras.nth(i).inner_text().strip() for i in range(paras.count())]
            desc_block = " ".join(desc_parts).strip()
        description = re.sub(r"^Description\s*", "", desc_block).strip()
    except:
        description = ""

    row_data = [algo_name, description]
    # ------- 先尝试 Parameters -------
    param_rows = page.locator("table.parameters tr:has(td)")
    if param_rows.count() > 0:
        params_info = []
        for i in range(param_rows.count()):
            row = param_rows.nth(i)
            tds = row.locator("td")
            if tds.count() < 2:
                continue

            raw_name = tds.nth(0).inner_text().strip()
            name = clean_name(raw_name)
            ptype = tds.nth(1).inner_text().strip()

            # 尝试获取 detail
            anchor_el = tds.nth(0).locator("a")
            anchor = anchor_el.get_attribute("href") if anchor_el.count() > 0 else None
            detail_text = ""

            if anchor and anchor.startswith("#"):
                param_id = anchor[1:]
                detail_text = page.evaluate(
                    """({paramId, paramName}) => {
                        let header = paramId ? document.querySelector(`p.pciParamName[id="${paramId}"]`) : null;
                        if (!header) {
                            const headers = Array.from(document.querySelectorAll('p.pciParamName'));
                            header = headers.find(h => {
                                let t = h.innerText.trim().split(':')[0].trim().toLowerCase();
                                return t === paramName.toLowerCase();
                            });
                        }
                        if (!header) return "";

                        const content = [];
                        let sibling = header.nextElementSibling;
                        while (sibling) {
                            if (sibling.matches('p.pciParamName')) break;
                            if (sibling.innerText) content.push(sibling.innerText.trim());
                            sibling = sibling.nextElementSibling;
                        }
                        return content.join(' ').replace(/\\s+/g, ' ').trim();
                    }""",
                    {"paramId": param_id, "paramName": name}
                )

                detail_text = re.sub(r"^\s*" + re.escape(name) + r"\s*", "", detail_text, flags=re.IGNORECASE).strip()

            params_info.extend([name, ptype, detail_text])

        return row_data + params_info

    # ------- 如果没有 Parameters，就尝试 Ports -------
    port_rows = page.locator("div.section#reference2228__sec_ports + div.tablenoborder table tbody tr.row")
    if port_rows.count() > 0:
        ports_info = []
        for i in range(port_rows.count()):
            row = port_rows.nth(i)
            try:
                name = row.locator("td").nth(1).inner_text().strip()
                port_type = row.locator("td").nth(3).inner_text().strip()
                port_detail_id = row.locator("td").nth(1).locator("a").get_attribute("href").replace("#", "")
                port_detail = page.locator(f"p#{port_detail_id} + p").inner_text().strip()
                ports_info.extend([name, port_type, port_detail])
            except:
                continue
        return row_data + ports_info

    # ------- 如果啥都没有，就只返回 名称+描述 -------
    return row_data


def main():
    with sync_playwright() as p:
        # 读取所有 URL
        with open(INPUT_URLS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            urls = [row["url"] for row in reader]

        # 写入结果
        with open(OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for url in urls:
                #print(f"正在处理: {url}")
                browser = None
                try:
                    browser = p.chromium.launch(headless=False)
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    )
                    page = context.new_page()

                    row_data = extract_algorithm_data(page, url)
                    writer.writerow(row_data)

                    #print(f"✅ 成功: {row_data[0]}, 数据列数: {len(row_data)}")

                except Exception as e:
                    print(f"❌ Failure: {url} | Error: {str(e)}")
                    writer.writerow(["Error", url, str(e)])

                finally:
                    if browser:
                        browser.close()

    print(f"Data has been written:  {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()
