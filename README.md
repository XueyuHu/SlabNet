# SlabNet

SlabNet is a surface-science workflow repository for building slab models, searching adsorbed intermediates, screening relaxed structures with machine-learned potentials, and connecting those structures into reaction networks.

## Focus

- chemically informed adsorbate placement on known slab surfaces
- rapid PES screening with pretrained `matgl` / `CHGNet` models
- retention of all reasonable local minima, not only the single lowest-energy structure
- reaction-network construction from retained intermediates
- structural energy-profile generation for complete and partial pathways

## Included module

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

