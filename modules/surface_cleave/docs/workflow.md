# Workflow Notes

## Goal

Starting from a bulk structure, enumerate low-index cleaved slabs and keep all terminations that may later serve as surface models for adsorption screening.

## Design choices

- use `pymatgen.core.surface.generate_all_slabs`
- default to `max_index = 2`
- save every returned termination
- write both a visualization-friendly CIF and a relax-ready VASP file
- freeze the bottom half of each slab so it can be passed downstream into slab-relaxation or adsorbate-screening workflows

## Suggested next step

For any chosen cleaved slab:

1. inspect the generated `slab.cif`
2. select a termination of interest from `summary.csv`
3. use `slab_relax_ready.vasp` as a starting point for slab relaxation
4. feed the relaxed slab into the intermediate-search workflow

