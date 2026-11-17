# Explanation guide

A walkthrough of the project's results, phase by phase, for reviewing the work without running it.

| File | What it is |
|---|---|
| `RDLD_Results_Walkthrough.ipynb` | The notebook, with every output stored. It opens directly on GitHub. |
| `RDLD_Results_Walkthrough.pdf` | The same notebook as a PDF report. |
| `assets/pipeline.png` | The pipeline diagram (Fig. 1 of the paper). |

For each phase the notebook shows who did it, the key part of the code, and the tables and figures that phase
produced. The work itself was done by the scripts in `scripts/`. The notebook only reads their saved results.

To re-run the notebook, put the data folders next to the repository (`../RDLD_data`, `../RDLD_data_test` and the two
`*_docboot` folders, see `docs/PIPELINE.md`) and run all cells. It needs pandas, numpy, matplotlib and Pillow, and it
takes a few seconds with no GPU.
