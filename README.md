# SlabNet

SlabNet is a surface-science workflow repository for building slab models, searching adsorbed intermediates, screening relaxed structures with machine-learned potentials, and connecting those structures into reaction networks.

## Focus

- chemically informed adsorbate placement on known slab surfaces
- rapid PES screening with pretrained `matgl` / `CHGNet` models
- retention of all reasonable local minima, not only the single lowest-energy structure
- reaction-network construction from retained intermediates
- structural energy-profile generation for complete and partial pathways

## Included modules

### `modules/surface_cleave`

This module starts from a bulk structure and enumerates low-index slab terminations for later surface studies.

- reads the bulk structure from `surface_cleave/POSCAR.cif` by default
- generates low-index slabs up to a chosen Miller cutoff
- saves every returned termination
- writes both `slab.cif` and a bottom-fixed `slab_relax_ready.vasp`
- records polarity, symmetry, and termination labels in a summary table

Example commands:

```bash
python -m modules.surface_cleave generate
python -m modules.surface_cleave overview
python -m modules.surface_cleave list
```

Current packaged cleave snapshot for `BaCoO3`:

- generated slab terminations: `6`
- unique low-index surfaces: `6`
- output summary: `modules/surface_cleave/results/generated_slabs/summary.csv`

### `modules/intermediate_search`

This module packages the intermediate-search workflow developed in this project as a reusable, documented unit.

- input slab bundled in the module
- search scripts and completion-search scripts included
- curated reasonable structures preserved for visualization
- merged reaction network and energy profiles included
- command-line entry available through `python -m modules.intermediate_search`

Module entry points:

```bash
python -m modules.intermediate_search overview
python -m modules.intermediate_search paths --status complete
python -m modules.intermediate_search describe
python -m modules.intermediate_search files
```

## Current intermediate-search snapshot

- input slab: `modules/intermediate_search/inputs/POSCAR.cif`
- merged reasonable structures: `56`
- reaction nodes: `47`
- reaction edges: `126`
- maximal `OO*`-origin pathways: `76`
- complete `OO* -> ... -> H2O*` pathways: `19`
- partial pathways retained as reasonable dead ends or side branches: `57`

## Key assets

- cleave module overview: `modules/surface_cleave/README.md`
- cleave summary table: `modules/surface_cleave/results/generated_slabs/summary.csv`
- module overview: `modules/intermediate_search/README.md`
- workflow notes: `modules/intermediate_search/docs/workflow.md`
- merged pathway list: `modules/intermediate_search/results/merged_network/all_paths.txt`
- merged energy-profile figure: `modules/intermediate_search/results/merged_network/energy_profiles/all_paths_profiles.png`
- merged reasonable-structure manifest: `modules/intermediate_search/results/reasonable_structures_merged/manifest.csv`

## Repository philosophy

This repository favors packaging a workflow in a way that is both:

- reusable for future slab systems
- inspectable for current scientific results

That means the repo keeps:

- scripts
- workflow notes
- curated summaries
- curated reasonable structures
- packaged network/profile outputs

and avoids committing heavyweight caches, model stores, and unrelated training artifacts.
