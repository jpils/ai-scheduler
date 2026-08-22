import sys
from ase.io import read, write


def sort_atoms_for_vasp(atoms):
    """
    VASP POSCAR requires one species entry per contiguous species block.

    ASE preserves input atom order by default; LAMMPS dumps often interleave
    species (K Ta O K Ta O ...), which makes ASE write repeated species labels
    and VASP rejects the POSCAR. Keep a stable order with common KTaO3 species
    first, then any remaining species alphabetically.
    """

    preferred = {"K": 0, "Ta": 1, "O": 2}
    symbols = atoms.get_chemical_symbols()
    indices = sorted(
        range(len(atoms)),
        key=lambda i: (preferred.get(symbols[i], 100), symbols[i], i),
    )
    return atoms[indices]


def main():
    if len(sys.argv) < 3:
        print("Invalid arguments.", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "count":
        if len(sys.argv) != 3:
            print(
                "Usage: xyz_to_poscar.py count <input.xyz>",
                file=sys.stderr,
            )
            sys.exit(1)

        input_path = sys.argv[2]

        try:
            configs = read(input_path, index=":")
            print(len(configs))
        except Exception as e:
            print(f"ASE Counting Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif mode == "extract":
        if len(sys.argv) != 5:
            print(
                "Usage: xyz_to_poscar.py extract <input.xyz> <index> <output.POSCAR>",
                file=sys.stderr,
            )
            sys.exit(1)

        input_path = sys.argv[2]
        target_index = int(sys.argv[3])
        output_path = sys.argv[4]

        try:
            atoms = read(input_path, index=target_index)
            atoms = sort_atoms_for_vasp(atoms)
            write(output_path, atoms, format="vasp", vasp5=True, direct=False)
        except Exception as e:
            print(
                f"ASE Conversion Error at index {target_index}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    else:
        print(f"Unknown mode '{mode}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
