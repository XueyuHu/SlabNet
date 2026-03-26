# Intermediate Search Module

This module packages a complete workflow for searching, relaxing, filtering, and connecting surface intermediates on a known slab.

## What this module does

- starts from a slab structure in `inputs/POSCAR.cif`
- places chemically guided adsorbates such as `O*`, `OH*`, `H2O*`, `OO*`, `OOH*`, and `H2O2*`
- performs orientation sampling and CHGNet-MatPES-PBE relaxation
- filters out unreasonable structures using force and short-contact checks
- preserves all reasonable results for visualization
- builds reaction networks from `+O / -O / +H / -H` transformations
- generates energy profiles for both complete and partial pathways

## Directory layout

```text
modules/intermediate_search/
- docs/
- inputs/
  - POSCAR.cif
- results/
  - completion_search/
  - main_search/
  - merged_network/
  - reasonable_structures_merged/
- scripts/
```

## Included scripts

- `scripts/screen_chgnet_intermediates.py`
  First chemically guided screen for the original nine intermediate families.
- `scripts/oh_reaction_search_space.py`
  Defines the adsorbate states and `+O / -O / +H / -H` neighbor relations.
- `scripts/search_orr_intermediates.py`
  Expanded ORR-like search over multiple sites and bonding motifs.
- `scripts/search_missing_intermediates.py`
  Completion search for contexts that were not yet closed.
- `scripts/build_reaction_pathways.py`
  Builds the reaction graph from reasonable structures.
- `scripts/draw_reaction_graph.py`
  Draws the reaction-graph visualization.
- `scripts/build_merged_reaction_profiles.py`
  Merges main and completion results, enumerates maximal pathways, and writes energy profiles.

## Current packaged results

- merged reasonable structures: `56`
- merged network nodes: `47`
- merged network edges: `126`
- maximal `OO*`-origin pathways: `76`
- complete pathways ending at `H2O*`: `19`
- partial pathways ending at another reasonable intermediate: `57`

## Key result files

- `results/main_search/summary.csv`
- `results/completion_search/summary.csv`
- `results/merged_network/all_paths.txt`
- `results/merged_network/path_counts.json`
- `results/merged_network/picture_description.txt`
- `results/merged_network/energy_profiles/all_paths_profiles.png`
- `results/reasonable_structures_merged/manifest.csv`

## Reproduction notes

The workflow was run locally with:

- local `matgl` source tree
- `CHGNet-MatPES-PBE-2025.2.10-2.7M-PES`
- `ASE` + `FIRE`
- chemically guided placement rules where the anchoring atom is chosen by bonding role rather than pure geometry

Typical execution order:

1. `screen_chgnet_intermediates.py`
2. `search_orr_intermediates.py`
3. `search_missing_intermediates.py`
4. `build_reaction_pathways.py`
5. `draw_reaction_graph.py`
6. `build_merged_reaction_profiles.py`

## Notes on scope

This module intentionally keeps the curated intermediate-search assets and summaries that are useful for inspection and reuse:

- input structure
- workflow scripts
- summary tables
- merged reasonable structures
- merged network and energy profiles

It does not package the full raw training caches, pretrained-model weights, or every heavy exploratory staging directory.
