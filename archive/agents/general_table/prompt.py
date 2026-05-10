"""General table extraction prompt."""

GENERAL_TABLE_PROMPT = """Extract ALL tables from this scanned PDF page image.

INSTRUCTIONS:
- Return each table as a separate entry, in top-to-bottom page order
- Return headers as a list of strings; each row as a list of cell values
- Preserve text exactly (leading zeros, punctuation, symbols like ≥ † %)
- Each row must have the same number of values as there are headers
- Empty cells → empty string ""

SPANNING / MULTI-LEVEL HEADERS:
- If a header spans multiple sub-columns (e.g. "Drug (N=100)" over "n" and "(%)"), emit one header per sub-column
- Use the spanning header text for the first sub-column and "" for the rest
- Example headers: ["", "Atorvastatin", "", "Fluvastatin", ""] where each drug spans n and (%) sub-columns
- The second header row (sub-headers like "n", "(%)", "Mean", "(SE)") becomes the actual headers list

HIERARCHICAL / GROUP ROWS:
- Section headings or group labels that have no data → fill data cells with ""
- Keep the label in the first column so row structure is preserved

CONTINUATION:
- If a table continues beyond the page bottom, set table_appears_complete to false
- If fully contained, set table_appears_complete to true
- Return headers only if this page contains that table's header row
"""
