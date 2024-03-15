## Post-Processing Scripts

`awl2struct.py` is a script to transform the generated `L, W, A, X` to the `cif` format. 

`compute_metrics.py` is a script to calculate the structure and composition validity of the generated structures.

`compute_metrics_matbench.py` is a script to calculate the novelty and uniqueness of the generated structures.

`e_above_hull.py` is a script to calculate the energy above the hull of the generated structures based on the Materials Project database.

`matgl_relax.py` is a script to relax the generated structures using the `matgl` package.

`plot_embeddings.py` is a script to visualize correlation of the learned embeddings vectors of different elements.