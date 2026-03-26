# SlabNet

`SlabNet` is a workspace for surface-structure generation, adsorbate/intermediate screening, and reaction-pathway analysis on slab models.

This repo currently includes a curated intermediate-search module here:

- `modules/intermediate_search`

That module packages the workflow we built for:

- chemically guided adsorbate placement on a known slab
- CHGNet-MatPES-PBE relaxation screening
- reasonable-structure collection
- reaction-network construction
- energy-profile generation for complete and partial ORR-like pathways

## Current module snapshot

- input slab: `Intermediates/POSCAR.cif`
- merged reasonable structures: `56`
- merged reaction nodes: `47`
- merged reaction edges: `126`
- maximal `OO*`-origin pathways: `76`
- complete `OO* -> ... -> H2O*` pathways: `19`

## Main outputs

- module overview: `modules/intermediate_search/README.md`
- merged network summary: `modules/intermediate_search/results/merged_network`
- merged reasonable structures: `modules/intermediate_search/results/reasonable_structures_merged`
