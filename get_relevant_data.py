# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "beautifulsoup4",
# ]
# ///

from bs4 import BeautifulSoup


with open("scraped_content.html", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# Get all wikitables under the content
tables = soup.select("main#content table.wikitable")

# The first table is usually the target one (validate manually if needed)
target_table = tables[0]
rows = target_table.select("tr")[1:]  # skip header row
for row in rows:
    cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
    print(cols)