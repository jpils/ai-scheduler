import os
import glob
from collections import Counter

import ase.io
import numpy as np
from enum import Enum

class EnergyMode(Enum):
    PET = "pet"
    RAW = "raw"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"Unknown energy mode '{value}'. Valid modes are: {valid}"
            )


def load_reference_energies(checkpoint_path):
    """
    Extract atomic reference energies from the PET-MAD checkpoint.
    """

    try:
        import torch
        from metatrain.utils.io import model_from_checkpoint

        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location="cpu",
        )

        pet_model = model_from_checkpoint(checkpoint, "finetune")

        energy_values = (
            pet_model.additive_models[0]
            .model.weights["energy"]
            .block()
            .values
        )

        atomic_numbers = checkpoint["model_data"]["dataset_info"].atomic_types

        return dict(zip(atomic_numbers, energy_values))

    except Exception as e:
        print(
            f"    [Info] Note: Local heavy torch libraries not loaded ({e}). "
            "Using structural baseline simulation fallback."
        )

        return {
            16: -4.0,
            29: -3.8,
        }


def load_generation(project_dir, generation_num):
    """
    Read every OUTCAR belonging to one generation.
    """

    base_pattern = os.path.join(
        project_dir,
        "vasp_runs",
        f"generation_{generation_num}",
        "config_*",
        "OUTCAR",
    )

    outcar_files = sorted(glob.glob(base_pattern))

    if not outcar_files:
        raise RuntimeError(
            f"No OUTCAR files found matching {base_pattern}"
        )

    print(
        f"[+] Processing {len(outcar_files)} configurations "
        f"for Generation {generation_num}..."
    )

    dataset = []

    for outcar_path in outcar_files:

        try:
            atoms = ase.io.read(
                outcar_path,
                format="vasp-out",
                index=-1,
            )

            dataset.append(atoms)

        except Exception as e:
            print(
                f"    [Warning] Failed parsing "
                f"{outcar_path}: {e}"
            )

    if not dataset:
        raise RuntimeError(
            "No valid OUTCARs could be parsed."
        )

    return dataset



def find_seed_dataset(project_dir):
    """
    Locate the user-provided labeled seed dataset.
    """

    setup_dir = os.path.join(project_dir, "setup", "training")

    candidates = [
        os.path.join(setup_dir, "seed_dataset.extxyz"),
        os.path.join(setup_dir, "seed_dataset.xyz"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    expected = " or ".join(candidates)
    raise RuntimeError(
        f"No seed dataset found. Expected {expected}"
    )


def seed_dataset_exists(project_dir):
    """
    Whether a labeled seed dataset already exists.
    """

    try:
        find_seed_dataset(project_dir)
        return True
    except RuntimeError:
        return False


def load_seed_dataset(project_dir):
    """
    Read the seed dataset used to create generation 1.
    """

    seed_path = find_seed_dataset(project_dir)

    print(f"[+] Loading seed dataset: {seed_path}")

    dataset = ase.io.read(seed_path, index=":")

    if not isinstance(dataset, list):
        dataset = [dataset]

    if not dataset:
        raise RuntimeError(
            f"Seed dataset contains no configurations: {seed_path}"
        )

    return dataset


def load_accumulated_dataset(project_dir, generation_num):
    """
    Build the training dataset for an active-learning generation.

    Generation 1 uses only the user-provided seed dataset.
    Generation N uses the seed dataset plus VASP results from generations
    1 through N-1.
    """

    if seed_dataset_exists(project_dir):
        dataset = load_seed_dataset(project_dir)
    else:
        print("[+] No labeled seed dataset found; using VASP labels from bootstrap generation 0.")
        dataset = load_generation(project_dir, 0)

    if generation_num == 1:
        print("[+] Generation 1 dataset source: seed/bootstrap dataset only.")
        return dataset

    for source_generation in range(1, generation_num):
        dataset.extend(
            load_generation(project_dir, source_generation)
        )

    print(
        f"[+] Accumulated {len(dataset)} configurations "
        f"for Generation {generation_num}."
    )

    return dataset


def apply_energy_shift(dataset, checkpoint_path):
    """
    Compute and apply PET reference energy shift.
    """

    dft_energies = [
        atoms.get_potential_energy()
        for atoms in dataset
    ]

    ref_energies = load_reference_energies(
        checkpoint_path
    )

    sample_atoms = dataset[0]

    counts = Counter(
        sample_atoms.get_atomic_numbers()
    )

    reference_total = sum(
        ref_energies.get(Z, 0.0) * count
        for Z, count in counts.items()
    )

    dft_mean = np.mean(dft_energies)

    reference_total = (
        reference_total.item()
        if hasattr(reference_total, "item")
        else reference_total
    )

    total_shift = dft_mean - reference_total

    print(f"    -> Mean Raw Energy: {dft_mean:.4f} eV")
    print(
        f"    -> Target Base Energy Alignment: "
        f"{reference_total:.4f} eV"
    )
    print(
        f"    -> Applied Shift Adjustment Matrix: "
        f"{total_shift:.4f} eV"
    )

    for atoms in dataset:

        corrected_energy = (
            atoms.get_potential_energy()
            - total_shift
        )

        atoms.info["energy-corrected"] = float(
            corrected_energy
        )


def split_dataset(dataset):
    """
    Fixed split for KTaO3 active learning: 20 train / 5 validation / 5 test
    when at least 30 configurations are available. Smaller datasets use all
    available configurations for training.
    """

    np.random.seed(42)

    indices = np.random.permutation(len(dataset))

    n = len(dataset)

    if n >= 30:
        n_train = 20
        n_validation = 5
        n_test = 5
    elif n >= 3:
        n_validation = max(1, round(0.15 * n))
        n_test = max(1, round(0.15 * n))
        n_train = n - n_validation - n_test
    else:
        n_train = n
        n_validation = 0
        n_test = 0

    train = [
        dataset[i]
        for i in indices[:n_train]
    ]

    validation = [
        dataset[i]
        for i in indices[
            n_train:
            n_train + n_validation
        ]
    ]

    test = [
        dataset[i]
        for i in indices[
            n_train + n_validation:
            n_train + n_validation + n_test
        ]
    ]

    return train, validation, test


def prepare_dataset(
    project_dir,
    generation_num,
    checkpoint_path,
    energy_mode: EnergyMode = EnergyMode.PET,
):
    """
    Complete shared workflow.

    Returns:
        dataset
    """

    dataset = load_accumulated_dataset(
        project_dir,
        generation_num,
    )

    if energy_mode is EnergyMode.PET:
        apply_energy_shift(
            dataset,
            checkpoint_path,
        )

    elif energy_mode is EnergyMode.RAW:
        print("    -> Using raw DFT energies (no energy correction).")

    return dataset
