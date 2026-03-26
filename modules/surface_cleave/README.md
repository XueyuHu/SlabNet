# Surface Cleave Module

This module takes a bulk structure and generates low-index slabs with all terminations returned by the cleaving routine.

## What it does

- reads a bulk structure from `surface_cleave/POSCAR.cif` by default
- enumerates low-index surfaces up to a chosen `max_index`
- retains all slab terminations returned by the cleave search
- expands the in-plane cell to avoid overly small adsorption cells
- writes each slab as both `slab.cif` and `slab_relax_ready.vasp`
- prepares a relax-ready slab where the bottom half is fixed and the top half is free
- writes a summary table with Miller indices, termination labels, polarity, symmetry, and output paths
- can score slabs with the CHGNet MatPES PBE model and rank average surface energies

## CLI

```bash
python -m modules.surface_cleave generate
python -m modules.surface_cleave generate --score-model
python -m modules.surface_cleave overview
python -m modules.surface_cleave list
python -m modules.surface_cleave list --hkl 1,0,0
```

## Current packaged result

For the bundled `BaCoO3` bulk structure, the current generated library contains:

- `6` slab terminations
- `6` unique low-index Miller surfaces
- minimum lateral-size target: `8.0 A`
- minimum in-plane area target: `64.0 A^2`
- bundled summary table in `results/generated_slabs/summary.csv`
- CHGNet-ranked surface energies in the same summary table

## Outputs

- `results/generated_slabs/summary.csv`
- `results/generated_slabs/summary.json`
- `results/generated_slabs/bulk_reference.json`
- one folder per slab termination containing:
  - `slab.cif`
  - `slab_relax_ready.vasp`
  - `slab_relaxed.cif`
  - `energy.json`
  - `metadata.json`

## Notes

The relax-ready VASP file is designed to match the slab-screening workflow used in the intermediate-search module:

- bottom half fixed
- top half free for later relaxation and adsorption studies
