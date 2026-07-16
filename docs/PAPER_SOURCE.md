# Paper source

The only editable DRIP paper source is the nested Overleaf Git repository:

```text
/home/jyliu/RAG-project/overleaf-paper/main.tex
```

Section files live in `overleaf-paper/sections/`.  Do not create parallel paper
TeX files under the code repository's `docs/`, `sections/`, or `motivation/`
directories.  Markdown design notes and raw experiment JSON remain in the code
repository; paper-ready claims and tables belong in the Overleaf repository.

## Consolidation map

The previous paper fragments have been consolidated as follows:

| Previous location | Canonical Overleaf location |
| --- | --- |
| `motivation/motivation.tex` | `overleaf-paper/sections/1_introduction.tex` |
| root-level `sections/*.tex` drafts | `overleaf-paper/sections/*.tex` |
| `motivation/paper_figs/experiments/*.tex` | `overleaf-paper/sections/5_experiment.tex` |
| `docs/design/DRIP_TALK_BEAMER*.tex` | retired; the paper sections are now canonical |

Old QDC/QARC tables were not copied into the paper because they use obsolete
method names and evaluation protocols.  Their underlying experiment records and
Markdown notes remain in the code repository when historical comparison is
needed.  Standalone generated benchmark snippets under `benchmarks/archive_legacy/results/` are
benchmark artifacts, not editable paper sections.

Local compile:

```bash
cd /home/jyliu/RAG-project/overleaf-paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Before editing from another machine, run `git pull --rebase` inside
`overleaf-paper/`.  Commit and push paper changes from that nested repository,
not from the parent RAG code repository.
