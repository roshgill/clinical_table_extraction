"""Simple table extraction prompt."""

SIMPLE_TABLE_PROMPT = """Extract the table from this scanned PDF page image.

INSTRUCTIONS:
- Identify the column headers and all data rows
- Return headers as a list of strings and each row as a list of cell values
- Preserve all text exactly as it appears (including leading zeros, punctuation, spacing)
- Each row must have the same number of values as there are headers
- If a cell is empty, return an empty string for that cell
- If the table continues beyond the bottom of this page, set table_appears_complete to false
- If the table is fully contained on this page, set table_appears_complete to true
"""
