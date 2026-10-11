"""Static checks for the extension notebook; does not replace a full data-backed rerun."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Notebooks" / "modeling_extension.ipynb"


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"Notebook not found: {NOTEBOOK}", file=sys.stderr)
        return 2
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not parse notebook JSON: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []
    cells = notebook.get("cells", [])
    if not cells:
        errors.append("Notebook has no cells.")

    code_cells = 0
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = "".join(cell.get("source", []))
        try:
            compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"Cell {index} syntax error at line {exc.lineno}: {exc.msg}")
        if cell.get("outputs", []):
            errors.append(f"Cell {index} retains cached outputs; rerun before publishing.")
        if cell.get("execution_count") is not None:
            errors.append(f"Cell {index} retains an execution count; rerun before publishing.")

    source_text = "\n".join("".join(c.get("source", [])) for c in cells)
    prohibited_claims = [
        (r"strictly compliant with.{0,30}Regulation B", "unsupported regulatory compliance claim"),
        (r"Basel Foundation.?IRB", "unsupported retail Basel F-IRB framing"),
        (r"Adverse Action Notice Generation", "adverse-action notice wording"),
        (r"Framework v2\.1", "obsolete internal framework label"),
        (r"\[cite:", "placeholder citation"),
        (r"[A-Z]:\\\\Users\\\\[^\\n]*Credit_Risk_Project", "local user path"),
    ]
    for pattern, description in prohibited_claims:
        if re.search(pattern, source_text, flags=re.IGNORECASE):
            errors.append(f"Notebook contains {description}.")

    if code_cells < 25:
        errors.append(f"Unexpectedly few code cells: {code_cells}.")

    if errors:
        print("Notebook static checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Notebook JSON and Python syntax checks passed ({code_cells} code cells).")
    print("All code-cell outputs are cleared; a clean data-backed run is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
