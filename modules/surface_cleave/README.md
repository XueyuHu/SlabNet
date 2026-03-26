# Surface Cleave Module

This module takes a bulk structure and generates low-index slabs with all terminations returned by the cleaving routine.

## What it does

- reads a bulk structure from `surface_cleave/POSCAR.cif` by default
- enumerates low-index surfaces up to a chosen `max_index`
- retains all slab terminations returned by the cleave search
- writes each slab as both `slab.cif` and `slab_relax_ready.vasp`
- prepares a relax-ready slab where the bottom half is fixed and the top half is free
- writes a summary table with Miller indices, termination labels, polarity, symmetry, and output paths

## CLI

```bash
python -m modules.surface_cleave generate
python -m modules.surface_cleave overview
python -m modules.surface_cleave list
python -m modules.surface_cleave list --hkl 1,0,0
```

## Current packaged result

For the bundled `BaCoO3` bulk structure, the current generated library contains:

- `6` slab terminations
- `6` unique low-index Miller surfaces
- bundled summary table in `results/generated_slabs/summary.csv`

## Outputs

- `results/generated_slabs/summary.csv`
- `results/generated_slabs/summary.json`
- one folder per slab termination containing:
  - `slab.cif`
  - `slab_relax_ready.vasp`
  - `metadata.json`

## Notes

The relax-ready VASP file is designed to match the slab-screening workflow used in the intermediate-search module:

- bottom half fixed
- top half free for later relaxation and adsorption studies
