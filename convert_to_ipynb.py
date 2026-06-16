import nbformat

nb = nbformat.v4.new_notebook()

with open('demo_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

cells = code.split('# %%')
for c in cells:
    c = c.strip()
    if not c:
        continue
    if c.startswith('[markdown]'):
        markdown_text = c.replace('[markdown]', '', 1).strip()
        lines = markdown_text.split('\n')
        # Remove comment markers
        cleaned_lines = [line[2:] if line.startswith('# ') else (line[1:] if line.startswith('#') else line) for line in lines]
        nb.cells.append(nbformat.v4.new_markdown_cell('\n'.join(cleaned_lines)))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(c))

with open('notebook_finalise.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
