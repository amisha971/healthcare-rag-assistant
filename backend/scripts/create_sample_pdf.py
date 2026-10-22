"""Create a synthetic PDF for manual Swagger testing. Run from backend/."""

from pathlib import Path

import pymupdf

target = Path("samples/synthetic-policy.pdf")
target.parent.mkdir(parents=True, exist_ok=True)
with pymupdf.open() as document:
    page = document.new_page()
    page.insert_text((72, 72), "SYNTHETIC EXAMPLE: Fictional Hospital Policy")
    page.insert_text((72, 100), "Staff must record equipment inspections in the daily checklist.")
    document.new_page()  # Blank page demonstrates preservation of page numbering.
    page = document.new_page()
    page.insert_text((72, 72), "SYNTHETIC EXAMPLE: Equipment Workflow")
    page.insert_text((72, 100), "Report missing inspection entries to the shift coordinator.")
    document.save(target)
print(f"Created {target.resolve()}")
