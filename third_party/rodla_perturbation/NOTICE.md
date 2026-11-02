Vendored unchanged from the RoDLA repository (Chen et al., CVPR 2024),
`perturbation/` directory. Licence: see LICENSE in this folder.

Used by this project: blur.py, noise.py, content.py, inconsistency.py (image-only
degradations). Not used: spatial.py and apply_perturbations.py (need detectron2;
spatial degradations move boxes and are excluded by design).

The degradation functions are kept unchanged so that single-degradation results
stay comparable with RoDLA's published numbers. Composition is our extension and
lives outside this folder.
