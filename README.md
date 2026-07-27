## Development usage

clone repo 

nix develop

cargo run --manifest-path path/to/Cargo.toml

## Installation

clone repo

nix develop

cargo install --path .

cargo run -- init

al_scheduler

## Configuration

Each project needs a `setup/config.toml` file. A minimal UPET mock-disagreement setup looks like:

```toml
[project]
generations = 2

[training]
backend = "upet"
energy_mode = "pet"

[committee]
members = 4

[disagreement]
mode = "mock"
max_selected = 8
min_rrmse = 0.02
max_rrmse = 0.40
```

For n2p2, use:

```toml
[training]
backend = "n2p2"
energy_mode = "raw"
```

`project.generations` controls how many active-learning generations are prepared.

`training.backend` selects the model backend. Supported values are `upet` and `n2p2`.

`training.energy_mode` controls the training energy target. `pet` applies the PET/foundation-model energy alignment used by UPET. `raw` uses raw DFT energies, which is the current n2p2 path.

`committee.members` sets how many committee models are prepared per generation.

`disagreement.mode = "mock"` is the current stable branch mode. It computes deterministic mock RRMSE-style force disagreement, writes `disagreement/generation_N/scores.csv`, and exports filtered structures to `selected_structures/generation_N.xyz`.

`max_selected`, `min_rrmse`, and `max_rrmse` define the filtering window. Frames below `min_rrmse` are treated as too certain; frames above `max_rrmse` are treated as too suspicious or unphysical for this first selection pass.
