"""DocLayNet class schema and COCO loading."""
import json

from rdld.config import ANNOTATIONS

# DocLayNet's 11 classes; category_id is 1-indexed in our COCO file
CATEGORIES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
    9: "Table", 10: "Text", 11: "Title",
}
NAME_TO_ID = {v: k for k, v in CATEGORIES.items()}

DOC_CATEGORIES = [
    "financial_reports", "scientific_articles", "government_tenders",
    "laws_and_regulations", "manuals", "patents",
]


def load_coco(path=ANNOTATIONS):
    return json.loads(path.read_text(encoding="utf-8"))
