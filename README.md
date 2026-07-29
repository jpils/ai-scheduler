


<p align="center"><img width="500" height="500" alt="ChatGPT Image Jul 22, 2026, 04_42_34 PM" src="https://github.com/user-attachments/assets/e7519980-ca25-45d7-a14d-933f7ee56467" /></p>

## Development usage

clone repo 

nix develop

cargo run --manifest-path path/to/Cargo.toml

## Installation

clone repo

nix develop

cargo install --path .

alchemist --init

alchemist

When running from source without installing, pass binary flags after Cargo's
separator:

```bash
cargo run -- --init
```

Alchemist stores runtime Python files, `pixi.toml`, `pixi.lock`, and Pixi
environments in an Alchemist home directory. By default this is the platform
data directory, e.g. `~/.local/share/alchemist` on Linux.

Users can choose a different location by setting `ALCHEMIST_HOME` before
initialization and before later runs:

```bash
export ALCHEMIST_HOME=/data/fs201232/$USER/alchemist
cargo install --path .
alchemist --init
```

This is useful on clusters where the home directory has a small quota. If
`PIXI_CACHE_DIR`, `UV_CACHE_DIR`, or `XDG_CACHE_HOME` are not already set,
Alchemist gives Pixi default cache locations under `$ALCHEMIST_HOME/cache/`.

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

# Optional. Defaults to "slurm".
[execution]
runner = "dry-run" # "slurm", "local", or "dry-run"
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

`execution.runner` controls the execution backend. `slurm` is the default: it submits Slurm steps and also allows explicitly local steps such as mock disagreement. `local` only accepts steps that complete locally. `dry-run` validates/prepares work in a temporary project copy and simulates completion.
 
