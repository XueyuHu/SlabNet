# Workflow Notes

## Goal

Given a known slab, enumerate chemically plausible intermediates, relax them with a pretrained CHGNet-MatPES-PBE potential, retain all reasonable local minima, and build possible reaction pathways from those retained structures.

## Screening logic

1. Detect top-layer surface sites from the input slab.
2. Define chemically meaningful adsorption anchors.
3. Place intermediates with the bonding atom facing the expected surface site.
4. Rotate and tilt adsorbates to sample multiple local basins.
5. Relax candidates with CHGNet.
6. Reject structures with large residual force or unphysical short contacts.
7. Keep all reasonable structures, not just the lowest-energy member of each family.

## Adsorbate logic

- `OH*`: O anchors to cationic metal sites and H points away from the surface.
- `H2O*`: O anchors to the surface and H atoms remain outward.
- `OO*`: both end-on and side-on motifs are sampled.
- `OOH*` and `H2O2*`: custom templates are used, then rotated.

## Network logic

The reaction graph is built from local transformation rules:

- `+O`
- `-O`
- `+H`
- `-H`

Two structures are connected only when they share the same surface context and their adsorbate states differ by one allowed transformation.

## Packaged outputs

- `results/reasonable_structures_merged`
  Curated structures kept for visualization and reuse.
- `results/merged_network`
  Merged graph, path lists, reaction counts, and image descriptions.
- `results/merged_network/energy_profiles`
  Combined profile and per-path profile figures and CSV tables.

