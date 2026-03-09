# Appendix Assets

Generated files for direct inclusion into your LaTeX paper.

## Build

```powershell
"c:/Users/Chase/Downloads/Seminar QF/.venv/Scripts/python.exe" paper_appendix/build_appendix_assets.py
```

## Use in paper

1. Ensure your preamble includes:

```latex
\usepackage{float}
\usepackage{booktabs}
\usepackage{graphicx}
```

2. At the appendix location in your main `.tex`:

```latex
\input{paper_appendix/appendix_include}
```

3. In Overleaf, upload the full `paper_appendix/` folder (including `figures/*.png`).

## Generated outputs

- `paper_appendix/appendix_include.tex`
- `paper_appendix/tables/*.tex`
- `paper_appendix/figures/*.tex` and copied figure PNGs
