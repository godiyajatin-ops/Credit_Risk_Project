"""Static notebook checks; these do not replace a full data-backed rerun."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    (ROOT / "Notebooks" / "credit_risk_analysis.ipynb", False),
    (ROOT / "Notebooks" / "modeling_extension.ipynb", True),
]
LOCAL_PATH_PATTERN = re.compile(r"[A-Z]:\\\\Users\\\\[^\n]*Credit_Risk_Project", re.IGNORECASE)
PROHIBITED_CLAIMS = [
    (re.compile(r"strictly compliant with.{0,30}Regulation B", re.IGNORECASE), "unsupported regulatory compliance claim"),
    (re.compile(r"Basel Foundation.?IRB", re.IGNORECASE), "unsupported retail Basel F-IRB framing"),
    (re.compile(r"Adverse Action Notice Generation", re.IGNORECASE), "adverse-action notice wording"),
    (re.compile(r"Framework v2\.1", re.IGNORECASE), "obsolete internal framework label"),
    (re.compile(r"\[cite:", re.IGNORECASE), "placeholder citation"),
]


def main() -> int:
    errors: list[str] = []
    summaries: list[str] = []

    for notebook_path, require_clean_outputs in NOTEBOOKS:
        if not notebook_path.exists():
            errors.append(f"Notebook not found: {notebook_path.relative_to(ROOT)}")
            continue
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Could not parse {notebook_path.name}: {exc}")
            continue

        cells = notebook.get("cells", [])
        if not cells:
            errors.append(f"{notebook_path.name} has no cells.")
            continue

        code_cells = 0
        all_source = []
        all_outputs = []
        for index, cell in enumerate(cells):
            source = "".join(cell.get("source", []))
            all_source.append(source)
            outputs_json = json.dumps(cell.get("outputs", []), ensure_ascii=False)
            all_outputs.append(outputs_json)

            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            try:
                compile(source, f"{notebook_path.name}:cell_{index}", "exec")
            except SyntaxError as exc:
                errors.append(f"{notebook_path.name} cell {index} syntax error at line {exc.lineno}: {exc.msg}")

            if require_clean_outputs:
                if cell.get("outputs", []):
                    errors.append(f"{notebook_path.name} cell {index} retains cached outputs; rerun before publishing.")
                if cell.get("execution_count") is not None:
                    errors.append(f"{notebook_path.name} cell {index} retains an execution count; rerun before publishing.")

        source_text = "\n".join(all_source)
        output_text = "\n".join(all_outputs)
        for pattern, description in PROHIBITED_CLAIMS:
            if pattern.search(source_text):
                errors.append(f"{notebook_path.name} contains {description}.")
        if LOCAL_PATH_PATTERN.search(source_text) or LOCAL_PATH_PATTERN.search(output_text):
            errors.append(f"{notebook_path.name} contains a hard-coded local Windows path.")
        if notebook_path.name == "modeling_extension.ipynb" and code_cells < 25:
            errors.append(f"{notebook_path.name} has unexpectedly few code cells: {code_cells}.")
        summaries.append(f"{notebook_path.name}: {len(cells)} cells, {code_cells} code cells")

    if errors:
        print("Notebook static checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Notebook structure and Python syntax checks passed:")
    for summary in summaries:
        print(f"- {summary}")
    print("Extension notebook outputs are cleared; a clean data-backed run is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
