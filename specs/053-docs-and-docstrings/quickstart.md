# Quickstart: Document code and write project README

## Run the docstring audit

```bash
python -c "
import ast
from pathlib import Path

missing = []
for f in sorted(Path('src').rglob('*.py')):
    if f.name == '__init__.py' and f.stat().st_size == 0:
        continue  # empty package markers, not undocumented code
    tree = ast.parse(f.read_text(encoding='utf-8'))
    if not ast.get_docstring(tree):
        missing.append((str(f), 'module'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
            if not ast.get_docstring(node):
                missing.append((str(f), node.name))
print(missing)
"
```

## Expected outcome

- Before this spec's fixes: `[('src\\\\api\\\\main.py', 'health'),
  ('src\\\\api\\\\main.py', 'predict'),
  ('src\\\\app\\\\streamlit_app.py', 'get_model_and_vectorizer')]`
- After: `[]` (empty list — everything documented)

## Validate

```bash
python -m pytest -q
```

Full suite must still pass unmodified — this task changes no behavior,
only docstrings and `README.md`.
