import argparse
import gc
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from ase.data import chemical_symbols
from ase.io import iread, write


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate committee force disagreement on one MD trajectory."
    )
    parser.add_argument("--backend", choices=["upet", "n2p2"], required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--committee-models", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-structures", required=True)
    parser.add_argument("--max-selected", type=int, default=8)
    parser.add_argument("--min-rrmse", type=float, default=0.02)
    parser.add_argument("--max-rrmse", type=float, default=0.40)
    parser.add_argument("--device", default=os.environ.get("ALCHEMIST_DEVICE", "cuda"))
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=int(os.environ.get("ALCHEMIST_QBC_FRAME_STRIDE", "1")),
        help="Evaluate every Nth trajectory frame for QBC selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_selected < 1:
        raise SystemExit("--max-selected must be greater than zero")

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be greater than zero")

    if not os.path.isfile(args.trajectory):
        raise SystemExit(f"Trajectory not found: {args.trajectory}")

    if not os.path.isdir(args.committee_models):
        raise SystemExit(f"Committee model directory not found: {args.committee_models}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.selected_structures), exist_ok=True)

    run_dir = os.path.dirname(args.trajectory)
    symbols = read_lammps_symbols(os.path.join(run_dir, "input.lmp"))
    model_dirs = find_member_dirs(args.committee_models)

    if args.backend == "upet" and os.environ.get("ALCHEMIST_UPET_QBC_ENGINE", "ase") == "lammps":
        run_lammps_upet_disagreement(args, symbols, model_dirs)
        return

    calcs = build_calculators(args.backend, model_dirs, args.device)

    scores = []
    selected_frames = []

    for frame_index, atoms in enumerate(iter_strided_frames(args.trajectory, args.frame_stride)):
        if symbols:
            atoms.set_chemical_symbols(symbols_for_frame(symbols, atoms))

        forces = []

        for calc_index, calc in enumerate(calcs):
            print(f"[+] Frame {frame_index}: evaluating member {calc_index}")
            atoms.calc = calc
            forces.append(np.asarray(atoms.get_forces(), dtype=float))
            atoms.calc = None

        forces = np.asarray(forces)
        rms_abs, rms_rel_mean = force_rrmse(forces)
        scores.append(
            {
                "frame_index": frame_index,
                "timestep": atoms.info.get("Time", atoms.info.get("timestep", frame_index)),
                "rms_abs": rms_abs,
                "rrmse_forces": rms_rel_mean,
                "selected": False,
            }
        )

        atoms.info["rms_abs_forces"] = rms_abs
        atoms.info["rrmse_forces"] = rms_rel_mean
        selected_frames.append(atoms.copy())

        clear_accelerator_cache()

    selected_indices = select_indices(
        scores,
        max_selected=args.max_selected,
        min_rrmse=args.min_rrmse,
        max_rrmse=args.max_rrmse,
    )

    for index in selected_indices:
        scores[index]["selected"] = True

    write_scores(os.path.join(args.output_dir, "scores.csv"), scores)

    selected = [selected_frames[index] for index in selected_indices]

    if not selected:
        raise SystemExit(
            "No frames passed the disagreement selection window. "
            "Adjust min_rrmse/max_rrmse or inspect scores.csv."
        )

    write(os.path.join(args.output_dir, "selected.xyz"), selected, format="extxyz")
    write(args.selected_structures, selected, format="extxyz")

    print(f"[+] Evaluated {len(scores)} frames with {len(calcs)} committee members (stride={args.frame_stride}).")
    print(f"[+] Selected {len(selected)} frames.")
    print(f"[+] Scores: {os.path.join(args.output_dir, 'scores.csv')}")
    print(f"[+] Selected structures: {args.selected_structures}")


def run_lammps_upet_disagreement(args, symbols, model_dirs):
    lmp = shutil.which("lmp")
    if not lmp:
        raise SystemExit("LAMMPS binary `lmp` not found in PATH for UPET/LAMMPS QBC")

    run_dir = os.path.dirname(args.trajectory)
    box = read_lammps_box(os.path.join(run_dir, "lmp.data"))

    scores = []
    selected_frames = []

    for frame_index, atoms in enumerate(iter_strided_frames(args.trajectory, args.frame_stride)):
        if symbols:
            atoms.set_chemical_symbols(symbols_for_frame(symbols, atoms))

        forces = []
        for calc_index, model_dir in enumerate(model_dirs):
            model_path = find_first_file(model_dir, ["model.pt", "mock_trained_model.pt"])
            if model_path.endswith("mock_trained_model.pt"):
                raise SystemExit(f"Refusing to run LAMMPS QBC with mock model: {model_path}")
            print(f"[+] Frame {frame_index}: evaluating member {calc_index} with LAMMPS/metatomic")
            forces.append(run_lammps_force_eval(lmp, atoms, symbols, box, model_path, args.device))

        forces = np.asarray(forces)
        rms_abs, rms_rel_mean = force_rrmse(forces)
        scores.append(
            {
                "frame_index": frame_index,
                "timestep": atoms.info.get("Time", atoms.info.get("timestep", frame_index)),
                "rms_abs": rms_abs,
                "rrmse_forces": rms_rel_mean,
                "selected": False,
            }
        )
        atoms.info["rms_abs_forces"] = rms_abs
        atoms.info["rrmse_forces"] = rms_rel_mean
        selected_frames.append(atoms.copy())

    selected_indices = select_indices(
        scores,
        max_selected=args.max_selected,
        min_rrmse=args.min_rrmse,
        max_rrmse=args.max_rrmse,
    )

    for index in selected_indices:
        scores[index]["selected"] = True

    write_scores(os.path.join(args.output_dir, "scores.csv"), scores)
    selected = [selected_frames[index] for index in selected_indices]

    if not selected:
        raise SystemExit(
            "No frames passed the disagreement selection window. "
            "Adjust min_rrmse/max_rrmse or inspect scores.csv."
        )

    write(os.path.join(args.output_dir, "selected.xyz"), selected, format="extxyz")
    write(args.selected_structures, selected, format="extxyz")

    print(f"[+] Evaluated {len(scores)} frames with {len(model_dirs)} LAMMPS/metatomic committee members (stride={args.frame_stride}).")
    print(f"[+] Selected {len(selected)} frames.")
    print(f"[+] Scores: {os.path.join(args.output_dir, 'scores.csv')}")
    print(f"[+] Selected structures: {args.selected_structures}")


def read_lammps_box(path):
    bounds = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ["xlo", "xhi"]:
                bounds["x"] = (float(parts[0]), float(parts[1]))
            elif len(parts) == 4 and parts[2:] == ["ylo", "yhi"]:
                bounds["y"] = (float(parts[0]), float(parts[1]))
            elif len(parts) == 4 and parts[2:] == ["zlo", "zhi"]:
                bounds["z"] = (float(parts[0]), float(parts[1]))
    if set(bounds) != {"x", "y", "z"}:
        raise SystemExit(f"Could not read orthogonal LAMMPS box from {path}")
    return bounds


def run_lammps_force_eval(lmp, atoms, symbols, box, model_path, device):
    with tempfile.TemporaryDirectory(prefix="alchemist-qbc-") as tmp:
        data_path = os.path.join(tmp, "frame.data")
        input_path = os.path.join(tmp, "in.force")
        forces_path = os.path.join(tmp, "forces.dump")
        write_lammps_frame_data(data_path, atoms, symbols, box)
        with open(input_path, "w", encoding="utf-8") as handle:
            handle.write(
                "units metal\n"
                "atom_style atomic\n"
                "boundary p p p\n"
                "read_data frame.data\n"
                f"pair_style metatomic {model_path} device {device}\n"
                "pair_coeff * * 19 73 8\n"
                "neighbor 2.0 bin\n"
                "neigh_modify every 1 delay 0 check yes\n"
                "dump f all custom 1 forces.dump id fx fy fz\n"
                "dump_modify f sort id first yes\n"
                "run 0\n"
            )
        result = subprocess.run(
            [lmp, "-in", input_path],
            cwd=tmp,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                "LAMMPS/metatomic force evaluation failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return read_lammps_forces(forces_path, len(atoms))


def write_lammps_frame_data(path, atoms, symbols, box):
    symbol_to_type = {symbol: index + 1 for index, symbol in enumerate(symbols)}
    masses = {"K": 39.0983, "Ta": 180.94788, "O": 15.999}
    positions = atoms.get_positions()
    atom_symbols = atoms.get_chemical_symbols()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("alchemist QBC frame\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(symbols)} atom types\n\n")
        handle.write(f"{box['x'][0]:.12f} {box['x'][1]:.12f} xlo xhi\n")
        handle.write(f"{box['y'][0]:.12f} {box['y'][1]:.12f} ylo yhi\n")
        handle.write(f"{box['z'][0]:.12f} {box['z'][1]:.12f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for symbol in symbols:
            handle.write(f"{symbol_to_type[symbol]} {masses.get(symbol, 1.0)} # {symbol}\n")
        handle.write("\nAtoms # atomic\n\n")
        for atom_id, (symbol, pos) in enumerate(zip(atom_symbols, positions), start=1):
            handle.write(
                f"{atom_id} {symbol_to_type[symbol]} "
                f"{pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}\n"
            )


def read_lammps_forces(path, atom_count):
    rows = []
    in_atoms = False
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            if in_atoms:
                if line.startswith("ITEM:"):
                    break
                parts = line.split()
                if len(parts) >= 4:
                    rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    if len(rows) != atom_count:
        raise SystemExit(f"Expected {atom_count} force rows in {path}, found {len(rows)}")
    rows.sort(key=lambda row: row[0])
    return np.asarray([[fx, fy, fz] for _, fx, fy, fz in rows], dtype=float)


def iter_strided_frames(trajectory, frame_stride):
    for source_index, atoms in enumerate(
        iread(trajectory, index=":", format="lammps-dump-text")
    ):
        if source_index % frame_stride == 0:
            atoms.info["source_frame_index"] = source_index
            yield atoms


def force_rrmse(forces):
    member_count = forces.shape[0]
    mean = forces.mean(axis=0)
    var_per_atom = ((forces - mean) ** 2).sum(axis=-1).mean(axis=0) / member_count
    rms_abs = math.sqrt(float(var_per_atom.mean()))
    mean_norm = float(np.linalg.norm(mean, axis=-1).mean())
    return rms_abs, rms_abs / max(mean_norm, 1.0e-12)


def select_indices(scores, max_selected, min_rrmse, max_rrmse):
    candidates = [
        score
        for score in scores
        if min_rrmse <= score["rrmse_forces"] <= max_rrmse
    ]
    candidates.sort(key=lambda score: score["rrmse_forces"], reverse=True)
    return sorted(score["frame_index"] for score in candidates[:max_selected])


def write_scores(path, scores):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("frame_index,timestep,rms_abs_forces,rrmse_forces,selected\n")

        for score in scores:
            handle.write(
                f"{score['frame_index']},{score['timestep']},"
                f"{score['rms_abs']:.10f},{score['rrmse_forces']:.10f},"
                f"{str(score['selected']).lower()}\n"
            )


def find_member_dirs(committee_models):
    members = [
        os.path.join(committee_models, name)
        for name in sorted(os.listdir(committee_models))
        if name.startswith("member_")
        and os.path.isdir(os.path.join(committee_models, name))
    ]

    if len(members) < 2:
        raise SystemExit(
            f"Need at least two committee member directories in {committee_models}"
        )

    return members


def build_calculators(backend, model_dirs, device):
    if backend == "upet":
        return [build_upet_calculator(model_dir, device) for model_dir in model_dirs]

    return [build_n2p2_calculator(model_dir) for model_dir in model_dirs]


def build_upet_calculator(model_dir, device):
    model_path = find_first_file(
        model_dir,
        ["model.pt", "mock_trained_model.pt"],
    )

    if model_path.endswith("mock_trained_model.pt"):
        raise SystemExit(
            f"Refusing to run real UPET disagreement with mock model: {model_path}"
        )

    calculator_classes = []

    import_errors = []

    try:
        from upet.calculator import UPETCalculator
        calculator_classes.append(UPETCalculator)
    except Exception as error:
        import_errors.append(f"upet.calculator.UPETCalculator: {type(error).__name__}: {error}")

    for module_name, class_names in [
        (
            "metatensor.torch.atomistic.ase_calculator",
            ["MetatensorCalculator"],
        ),
        (
            "metatomic.torch.ase_calculator",
            ["MetatomicCalculator", "MetatensorCalculator"],
        ),
    ]:
        try:
            module = __import__(module_name, fromlist=class_names)

            for class_name in class_names:
                try:
                    candidate = getattr(module, class_name)
                except AttributeError:
                    continue

                calculator_classes.append(candidate)
        except Exception as error:
            import_errors.append(f"{module_name}: {type(error).__name__}: {error}")
            continue


    if not calculator_classes:
        message = (
            "Could not import an UPET/metatomic ASE calculator. "
            "Install the UPET ASE interface or adapt build_upet_calculator()."
        )
        if import_errors:
            message += "\nImport attempts:\n" + "\n".join(import_errors)
        raise SystemExit(message)

    errors = []

    for calculator_class in calculator_classes:
        for kwargs in [
            {"model": model_path, "device": device},
            {"model_path": model_path, "device": device},
            {"model": model_path},
            {"model_path": model_path},
            {"path": model_path, "device": device},
            {"checkpoint": model_path, "device": device},
        ]:
            try:
                return calculator_class(**kwargs)
            except Exception as error:
                errors.append(f"{calculator_class.__name__}{kwargs}: {type(error).__name__}: {error}")

        try:
            return calculator_class(model_path)
        except Exception as error:
            errors.append(f"{calculator_class.__name__}({model_path}): {type(error).__name__}: {error}")

    raise SystemExit(
        "Found an UPET/metatomic ASE calculator, but could not construct it:\n"
        + "\n".join(errors)
    )


def build_n2p2_calculator(model_dir):
    for required in ["input.nn", "scaling.data"]:
        path = os.path.join(model_dir, required)

        if not os.path.isfile(path):
            raise SystemExit(f"Missing n2p2 calculator file: {path}")

    if not any(
        name.startswith("weights.") and name.endswith(".data")
        for name in os.listdir(model_dir)
    ):
        raise SystemExit(f"No weights.*.data files found in {model_dir}")

    try:
        from ase.calculators.n2p2 import N2P2
    except ImportError as error:
        raise SystemExit(
            "Could not import ASE's n2p2 calculator. "
            "Install the ASE n2p2 interface or adapt build_n2p2_calculator()."
        ) from error

    errors = []

    for kwargs in [{"directory": model_dir}, {"label": model_dir}]:
        try:
            return N2P2(**kwargs)
        except TypeError as error:
            errors.append(f"N2P2{kwargs}: {error}")

    raise SystemExit(
        "Found ASE's n2p2 calculator, but could not construct it:\n"
        + "\n".join(errors)
    )


def find_first_file(directory, names):
    for name in names:
        path = os.path.join(directory, name)

        if os.path.isfile(path):
            return path

    raise SystemExit(
        f"None of {', '.join(names)} found in committee member directory {directory}"
    )


def read_lammps_symbols(input_lmp):
    if not os.path.isfile(input_lmp):
        return []

    with open(input_lmp, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.split("#", 1)[0].strip()

            if not stripped.startswith("pair_coeff"):
                continue

            fields = stripped.split()

            if len(fields) < 4 or fields[1:3] != ["*", "*"]:
                continue

            return [symbol_from_token(token) for token in fields[3:]]

    return []


def symbol_from_token(token):
    if token.isdigit():
        atomic_number = int(token)

        if atomic_number <= 0 or atomic_number >= len(chemical_symbols):
            raise SystemExit(f"Invalid atomic number in pair_coeff: {token}")

        return chemical_symbols[atomic_number]

    return token


def symbols_for_frame(type_symbols, atoms):
    numbers = atoms.get_atomic_numbers()

    if len(numbers) != len(atoms):
        return atoms.get_chemical_symbols()

    symbols = []

    for number in numbers:
        type_index = int(number) - 1

        if type_index < 0 or type_index >= len(type_symbols):
            raise SystemExit(
                f"LAMMPS atom type {number} has no pair_coeff mapping "
                f"({type_symbols})"
            )

        symbols.append(type_symbols[type_index])

    return symbols


def clear_accelerator_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    gc.collect()


if __name__ == "__main__":
    main()
