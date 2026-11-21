# Paper figures

The four result figures of the paper and the script that draws them.

| File | Figure in the paper |
|---|---|
| `figures/fig1_example.pdf` | Fig. 2: one page under a single degradation and a pair |
| `figures/fig2_interaction.pdf` | Fig. 3: mean interaction per degradation pair |
| `figures/fig3_retention.pdf` | Fig. 4: mAP retention on singles and pairs, with the mRD rank |
| `figures/fig4_rq2.pdf` | Fig. 5: within-condition correlation with downstream quality |
| `make_figures.py` | Draws all four from the saved result CSVs |

The figures are vector PDFs (Matplotlib, Times New Roman at 8 pt). The page images inside Fig. 2 are kept at
about 470 DPI. The pipeline flowchart (Fig. 1 of the paper) is drawn in TikZ inside the paper's `main.tex`.

To redraw the figures, keep the data folders `RDLD_data` and `RDLD_data_test` next to the repository
(see `docs/PIPELINE.md`) and run:

```
python paper_figures/make_figures.py
```

It needs matplotlib, numpy, pandas and Pillow, takes a few seconds and needs no GPU. The output is the same
as the PDFs stored here.
