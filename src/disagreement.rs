use crate::job_template::render_template;
use crate::paths::scheduler_home;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub struct DisagreementSettings {
    pub max_selected: usize,
    pub bootstrap_max_selected: Option<usize>,
    pub min_rrmse: f64,
    pub max_rrmse: f64,
}

impl Default for DisagreementSettings {
    fn default() -> Self {
        Self {
            max_selected: 8,
            bootstrap_max_selected: None,
            min_rrmse: 0.02,
            max_rrmse: 0.40,
        }
    }
}

pub struct MockDisagreementResult {
    pub selected_count: usize,
    pub scores_path: PathBuf,
    pub selected_path: PathBuf,
}

pub struct DisagreementWorkspace;

#[derive(Clone)]
struct DumpFrame {
    timestep: i64,
    bounds: BoxBounds,
    atoms: Vec<DumpAtom>,
}

#[derive(Clone, Copy)]
struct BoxBounds {
    xlo_bound: f64,
    xhi_bound: f64,
    ylo_bound: f64,
    yhi_bound: f64,
    zlo_bound: f64,
    zhi_bound: f64,
    xy: f64,
    xz: f64,
    yz: f64,
}

impl BoxBounds {
    fn lattice(&self) -> [[f64; 3]; 3] {
        let xlo = self.xlo_bound - 0.0_f64.min(self.xy).min(self.xz).min(self.xy + self.xz);
        let xhi = self.xhi_bound - 0.0_f64.max(self.xy).max(self.xz).max(self.xy + self.xz);
        let ylo = self.ylo_bound - 0.0_f64.min(self.yz);
        let yhi = self.yhi_bound - 0.0_f64.max(self.yz);
        let zlo = self.zlo_bound;
        let zhi = self.zhi_bound;

        [
            [xhi - xlo, 0.0, 0.0],
            [self.xy, yhi - ylo, 0.0],
            [self.xz, self.yz, zhi - zlo],
        ]
    }

    fn fractional_to_cartesian(&self, xs: f64, ys: f64, zs: f64) -> [f64; 3] {
        let lattice = self.lattice();

        [
            xs * lattice[0][0] + ys * lattice[1][0] + zs * lattice[2][0],
            xs * lattice[0][1] + ys * lattice[1][1] + zs * lattice[2][1],
            xs * lattice[0][2] + ys * lattice[1][2] + zs * lattice[2][2],
        ]
    }
}

#[derive(Clone)]
struct DumpAtom {
    id: usize,
    atom_type: usize,
    position: [f64; 3],
}

struct FrameScore {
    frame_index: usize,
    timestep: i64,
    rms_abs_forces: f64,
    rrmse_forces: f64,
    selected: bool,
}

impl DisagreementWorkspace {
    pub fn evaluate_mock(
        project_dir: &Path,
        generation: u32,
        backend: &str,
        committee_members: usize,
        settings: DisagreementSettings,
    ) -> Result<MockDisagreementResult, String> {
        if committee_members < 2 {
            return Err("Disagreement needs at least two committee members.".to_string());
        }

        let run_dir = project_dir
            .join("md_runs")
            .join(format!("generation_{generation}"))
            .join("run_000");
        let trajectory = run_dir.join("traj.dump");

        if !trajectory.is_file() {
            create_mock_trajectory(&run_dir.join("lmp.data"), &trajectory)?;
        }

        let frames = read_lammps_dump(&trajectory)?;

        if frames.is_empty() {
            return Err(format!("No frames found in {}", trajectory.display()));
        }

        let species = match read_lammps_species(&run_dir.join("input.lmp")) {
            Ok(species) => species,
            Err(error) => {
                eprintln!(
                    " ⚠️  Could not read LAMMPS species for mock disagreement ({}); using fallback species.",
                    error
                );
                fallback_species()
            }
        };

        let disagreement_dir = project_dir
            .join("disagreement")
            .join(format!("generation_{generation}"));
        fs::create_dir_all(&disagreement_dir).map_err(|error| {
            format!(
                "Failed to create disagreement directory {}: {}",
                disagreement_dir.display(),
                error
            )
        })?;

        let mut scores = Vec::new();

        for (frame_index, frame) in frames.iter().enumerate() {
            let (rms_abs_forces, rrmse_forces) =
                mock_force_disagreement(frame, frame_index, backend, committee_members);

            scores.push(FrameScore {
                frame_index,
                timestep: frame.timestep,
                rms_abs_forces,
                rrmse_forces,
                selected: false,
            });
        }

        let mut selected_indices = select_frames(&scores, settings);

        for score in &mut scores {
            score.selected = selected_indices.contains(&score.frame_index);
        }

        selected_indices.sort_unstable();

        let scores_path = disagreement_dir.join("scores.csv");
        write_scores(&scores_path, &scores)?;

        let local_selected = disagreement_dir.join("selected.xyz");
        write_selected_xyz(&local_selected, &frames, &selected_indices, &species)?;

        let selected_dir = project_dir.join("selected_structures");
        fs::create_dir_all(&selected_dir).map_err(|error| {
            format!(
                "Failed to create selected structures directory {}: {}",
                selected_dir.display(),
                error
            )
        })?;

        let selected_path = selected_dir.join(format!("generation_{generation}.xyz"));
        fs::copy(&local_selected, &selected_path).map_err(|error| {
            format!(
                "Failed to copy selected structures into {}: {}",
                selected_path.display(),
                error
            )
        })?;

        Ok(MockDisagreementResult {
            selected_count: selected_indices.len(),
            scores_path,
            selected_path,
        })
    }

    pub fn create_real_job_script(
        project_dir: &Path,
        setup_dir: &Path,
        generation: u32,
        backend: &str,
        settings: DisagreementSettings,
    ) -> Result<PathBuf, String> {
        let generation_dir = project_dir
            .join("disagreement")
            .join(format!("generation_{generation}"));

        fs::create_dir_all(&generation_dir).map_err(|error| {
            format!(
                "Failed to create disagreement directory {}: {}",
                generation_dir.display(),
                error
            )
        })?;

        let trajectory = project_dir
            .join("md_runs")
            .join(format!("generation_{generation}"))
            .join("run_000")
            .join("traj.dump");

        let committee_models = project_dir
            .join("md_runs")
            .join(format!("generation_{generation}"))
            .join("committee_models");

        let selected_structures = project_dir
            .join("selected_structures")
            .join(format!("generation_{generation}.xyz"));

        if let Some(parent) = selected_structures.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "Failed to create selected structures directory {}: {}",
                    parent.display(),
                    error
                )
            })?;
        }

        let project_python_script = project_dir
            .join("external")
            .join("alchemist")
            .join("python")
            .join("committee_disagreement.py");
        let installed_python_script = scheduler_home()
            .map_err(|error| error.to_string())?
            .join("python")
            .join("committee_disagreement.py");
        let python_script = if project_python_script.is_file() {
            project_python_script
        } else {
            installed_python_script
        };

        if !python_script.is_file() {
            return Err(format!(
                "Missing disagreement Python script: {}",
                python_script.display()
            ));
        }

        let template_path = setup_dir
            .join("jobscripts")
            .join(format!("{backend}_disagreement.sh.template"));

        if !template_path.is_file() {
            return Err(format!(
                "Missing disagreement job template: {}",
                template_path.display()
            ));
        }

        let script_path = generation_dir.join("submit_disagreement.sh");

        let max_selected = if generation == 0 {
            settings.bootstrap_max_selected.unwrap_or(settings.max_selected)
        } else {
            settings.max_selected
        };

        render_template(
            &template_path,
            &script_path,
            &[
                ("generation", generation.to_string()),
                ("project_dir", absolute_path_string(project_dir)?),
                ("python_script", absolute_path_string(&python_script)?),
                ("trajectory", absolute_path_string(&trajectory)?),
                ("committee_models", absolute_path_string(&committee_models)?),
                ("output_dir", absolute_path_string(&generation_dir)?),
                (
                    "selected_structures",
                    absolute_path_string(&selected_structures)?,
                ),
                ("max_selected", max_selected.to_string()),
                ("min_rrmse", settings.min_rrmse.to_string()),
                ("max_rrmse", settings.max_rrmse.to_string()),
            ],
        )
        .map_err(|error| {
            format!(
                "Failed to render disagreement job template {}: {}",
                template_path.display(),
                error
            )
        })?;

        set_executable(&script_path).map_err(|error| {
            format!(
                "Failed to mark disagreement job script executable {}: {}",
                script_path.display(),
                error
            )
        })?;

        Ok(script_path)
    }
}

fn read_lammps_dump(path: &Path) -> Result<Vec<DumpFrame>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {}", path.display(), error))?;
    let mut lines = text.lines().peekable();
    let mut frames = Vec::new();

    while let Some(line) = lines.next() {
        if line.trim() != "ITEM: TIMESTEP" {
            continue;
        }

        let timestep: i64 = parse_next(&mut lines, "timestep")?;
        expect_item(&mut lines, "ITEM: NUMBER OF ATOMS")?;
        let atom_count: usize = parse_next(&mut lines, "atom count")?;

        let bounds_header = lines
            .next()
            .ok_or_else(|| "Unexpected end of dump before box bounds".to_string())?;

        if !bounds_header.starts_with("ITEM: BOX BOUNDS") {
            return Err(format!("Expected BOX BOUNDS, found: {bounds_header}"));
        }

        let bounds = parse_bounds(&mut lines)?;
        let atoms_header = lines
            .next()
            .ok_or_else(|| "Unexpected end of dump before atom table".to_string())?;

        if !atoms_header.starts_with("ITEM: ATOMS") {
            return Err(format!("Expected ATOMS table, found: {atoms_header}"));
        }

        let columns: Vec<&str> = atoms_header.split_whitespace().skip(2).collect();
        let id_col = column_index(&columns, "id")?;
        let type_col = column_index(&columns, "type")?;
        let scaled_columns = match (
            column_index(&columns, "xs"),
            column_index(&columns, "ys"),
            column_index(&columns, "zs"),
        ) {
            (Ok(xs), Ok(ys), Ok(zs)) => Some((xs, ys, zs)),
            _ => None,
        };
        let cartesian_columns = match (
            column_index(&columns, "x"),
            column_index(&columns, "y"),
            column_index(&columns, "z"),
        ) {
            (Ok(x), Ok(y), Ok(z)) => Some((x, y, z)),
            _ => None,
        };

        if scaled_columns.is_none() && cartesian_columns.is_none() {
            return Err(
                "LAMMPS dump atom table must contain either xs/ys/zs or x/y/z columns".to_string(),
            );
        }

        let mut atoms = Vec::with_capacity(atom_count);

        for _ in 0..atom_count {
            let atom_line = lines
                .next()
                .ok_or_else(|| "Unexpected end of dump inside atom table".to_string())?;
            let fields: Vec<&str> = atom_line.split_whitespace().collect();

            let id = parse_field::<usize>(&fields, id_col, "atom id")?;
            let atom_type = parse_field::<usize>(&fields, type_col, "atom type")?;
            let position = if let Some((xs_col, ys_col, zs_col)) = scaled_columns {
                let xs = parse_field::<f64>(&fields, xs_col, "xs")?;
                let ys = parse_field::<f64>(&fields, ys_col, "ys")?;
                let zs = parse_field::<f64>(&fields, zs_col, "zs")?;

                bounds.fractional_to_cartesian(xs, ys, zs)
            } else if let Some((x_col, y_col, z_col)) = cartesian_columns {
                [
                    parse_field::<f64>(&fields, x_col, "x")?,
                    parse_field::<f64>(&fields, y_col, "y")?,
                    parse_field::<f64>(&fields, z_col, "z")?,
                ]
            } else {
                unreachable!();
            };

            atoms.push(DumpAtom {
                id,
                atom_type,
                position,
            });
        }

        atoms.sort_by_key(|atom| atom.id);

        frames.push(DumpFrame {
            timestep,
            bounds,
            atoms,
        });
    }

    Ok(frames)
}

fn create_mock_trajectory(data_path: &Path, trajectory_path: &Path) -> Result<(), String> {
    let (bounds, atoms) = match read_lammps_data(data_path) {
        Ok(data) => data,
        Err(error) => {
            eprintln!(
                " ⚠️  Could not read LAMMPS data for mock trajectory ({}); using fallback mock atoms.",
                error
            );
            fallback_lammps_data()
        }
    };
    let mut text = String::new();

    for frame_index in 0..11 {
        text.push_str("ITEM: TIMESTEP\n");
        text.push_str(&format!("{}\n", frame_index * 100));
        text.push_str("ITEM: NUMBER OF ATOMS\n");
        text.push_str(&format!("{}\n", atoms.len()));
        text.push_str("ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
        text.push_str(&format!(
            "{:.16e} {:.16e} {:.16e}\n",
            bounds.xlo_bound, bounds.xhi_bound, bounds.xy
        ));
        text.push_str(&format!(
            "{:.16e} {:.16e} {:.16e}\n",
            bounds.ylo_bound, bounds.yhi_bound, bounds.xz
        ));
        text.push_str(&format!(
            "{:.16e} {:.16e} {:.16e}\n",
            bounds.zlo_bound, bounds.zhi_bound, bounds.yz
        ));
        text.push_str("ITEM: ATOMS id type x y z\n");

        for atom in &atoms {
            let displacement = (frame_index as f64 * 0.01) * ((atom.id % 7) as f64 - 3.0);
            text.push_str(&format!(
                "{} {} {:.10} {:.10} {:.10}\n",
                atom.id,
                atom.atom_type,
                atom.position[0] + displacement,
                atom.position[1] - 0.5 * displacement,
                atom.position[2] + 0.25 * displacement
            ));
        }
    }

    fs::write(trajectory_path, text).map_err(|error| {
        format!(
            "Failed to write mock trajectory {}: {}",
            trajectory_path.display(),
            error
        )
    })
}

fn fallback_lammps_data() -> (BoxBounds, Vec<DumpAtom>) {
    let bounds = BoxBounds {
        xlo_bound: 0.0,
        xhi_bound: 10.0,
        ylo_bound: 0.0,
        yhi_bound: 10.0,
        zlo_bound: 0.0,
        zhi_bound: 10.0,
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };

    let atoms = (0..16)
        .map(|index| DumpAtom {
            id: index + 1,
            atom_type: if index % 2 == 0 { 1 } else { 2 },
            position: [
                1.0 + (index % 4) as f64 * 2.0,
                1.0 + ((index / 4) % 4) as f64 * 2.0,
                5.0,
            ],
        })
        .collect();

    (bounds, atoms)
}

fn read_lammps_data(path: &Path) -> Result<(BoxBounds, Vec<DumpAtom>), String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {}", path.display(), error))?;
    let mut xlo_bound = None;
    let mut xhi_bound = None;
    let mut ylo_bound = None;
    let mut yhi_bound = None;
    let mut zlo_bound = None;
    let mut zhi_bound = None;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yz = 0.0;
    let mut in_atoms = false;
    let mut atoms = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();

        if fields.len() >= 4 && fields[2] == "xlo" && fields[3] == "xhi" {
            xlo_bound = Some(parse_number(fields[0], "xlo")?);
            xhi_bound = Some(parse_number(fields[1], "xhi")?);
            continue;
        }

        if fields.len() >= 4 && fields[2] == "ylo" && fields[3] == "yhi" {
            ylo_bound = Some(parse_number(fields[0], "ylo")?);
            yhi_bound = Some(parse_number(fields[1], "yhi")?);
            continue;
        }

        if fields.len() >= 4 && fields[2] == "zlo" && fields[3] == "zhi" {
            zlo_bound = Some(parse_number(fields[0], "zlo")?);
            zhi_bound = Some(parse_number(fields[1], "zhi")?);
            continue;
        }

        if fields.len() >= 6 && fields[3] == "xy" && fields[4] == "xz" && fields[5] == "yz" {
            xy = parse_number(fields[0], "xy")?;
            xz = parse_number(fields[1], "xz")?;
            yz = parse_number(fields[2], "yz")?;
            continue;
        }

        if fields[0] == "Atoms" {
            in_atoms = true;
            continue;
        }

        if in_atoms && fields.len() >= 5 {
            atoms.push(DumpAtom {
                id: parse_number::<usize>(fields[0], "atom id")?,
                atom_type: parse_number::<usize>(fields[1], "atom type")?,
                position: [
                    parse_number::<f64>(fields[2], "x")?,
                    parse_number::<f64>(fields[3], "y")?,
                    parse_number::<f64>(fields[4], "z")?,
                ],
            });
        }
    }

    if atoms.is_empty() {
        return Err(format!(
            "No atoms found in LAMMPS data file {}",
            path.display()
        ));
    }

    Ok((
        BoxBounds {
            xlo_bound: xlo_bound.ok_or_else(|| "Missing xlo/xhi bounds".to_string())?,
            xhi_bound: xhi_bound.ok_or_else(|| "Missing xlo/xhi bounds".to_string())?,
            ylo_bound: ylo_bound.ok_or_else(|| "Missing ylo/yhi bounds".to_string())?,
            yhi_bound: yhi_bound.ok_or_else(|| "Missing ylo/yhi bounds".to_string())?,
            zlo_bound: zlo_bound.ok_or_else(|| "Missing zlo/zhi bounds".to_string())?,
            zhi_bound: zhi_bound.ok_or_else(|| "Missing zlo/zhi bounds".to_string())?,
            xy,
            xz,
            yz,
        },
        atoms,
    ))
}

fn parse_bounds<'a, I>(lines: &mut I) -> Result<BoxBounds, String>
where
    I: Iterator<Item = &'a str>,
{
    let x = parse_bound_line(lines.next(), "x bounds")?;
    let y = parse_bound_line(lines.next(), "y bounds")?;
    let z = parse_bound_line(lines.next(), "z bounds")?;

    Ok(BoxBounds {
        xlo_bound: x[0],
        xhi_bound: x[1],
        xy: x[2],
        ylo_bound: y[0],
        yhi_bound: y[1],
        xz: y[2],
        zlo_bound: z[0],
        zhi_bound: z[1],
        yz: z[2],
    })
}

fn parse_bound_line(line: Option<&str>, label: &str) -> Result<[f64; 3], String> {
    let line = line.ok_or_else(|| format!("Unexpected end of dump while reading {label}"))?;
    let values: Vec<f64> = line
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|error| format!("Invalid {label} value '{value}': {error}"))
        })
        .collect::<Result<_, _>>()?;

    if values.len() != 3 {
        return Err(format!(
            "Expected three values for {label}, found {}",
            values.len()
        ));
    }

    Ok([values[0], values[1], values[2]])
}

fn mock_force_disagreement(
    frame: &DumpFrame,
    frame_index: usize,
    backend: &str,
    committee_members: usize,
) -> (f64, f64) {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut components = 0usize;

    for (atom_index, atom) in frame.atoms.iter().enumerate() {
        for component in 0..3 {
            let mut predictions = Vec::with_capacity(committee_members);

            for member_index in 0..committee_members {
                predictions.push(mock_force_component(
                    frame_index,
                    atom_index,
                    atom,
                    component,
                    member_index,
                    backend,
                ));
            }

            let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
            let variance = predictions
                .iter()
                .map(|prediction| {
                    let diff = prediction - mean;
                    diff * diff
                })
                .sum::<f64>()
                / predictions.len() as f64;

            numerator += variance;
            denominator += mean * mean;
            components += 1;
        }
    }

    let rms_abs = (numerator / components as f64).sqrt();
    let rms_mean = (denominator / components as f64).sqrt();

    (rms_abs, rms_abs / rms_mean.max(1.0e-12))
}

fn mock_force_component(
    frame_index: usize,
    atom_index: usize,
    atom: &DumpAtom,
    component: usize,
    member_index: usize,
    backend: &str,
) -> f64 {
    let backend_shift = if backend == "n2p2" { 0.071 } else { 0.0 };
    let base = 0.35
        + backend_shift
        + (atom.position[component] * 0.13).sin()
        + ((atom_index + 1) as f64 * 0.017 + component as f64).cos() * 0.15;
    let model_offset = (member_index as f64 - 1.5) * 0.006;
    let uncertainty = ((frame_index + 1) as f64 * 0.11).sin().abs() * (member_index as f64 * 0.015);

    base + model_offset + uncertainty
}

fn select_frames(scores: &[FrameScore], settings: DisagreementSettings) -> Vec<usize> {
    let mut candidates: Vec<&FrameScore> = scores
        .iter()
        .filter(|score| {
            score.rrmse_forces >= settings.min_rrmse && score.rrmse_forces <= settings.max_rrmse
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.rrmse_forces
            .partial_cmp(&a.rrmse_forces)
            .unwrap_or(Ordering::Equal)
    });

    candidates
        .into_iter()
        .take(settings.max_selected)
        .map(|score| score.frame_index)
        .collect()
}

fn write_scores(path: &Path, scores: &[FrameScore]) -> Result<(), String> {
    let mut text = String::from("frame_index,timestep,rms_abs_forces,rrmse_forces,selected\n");

    for score in scores {
        text.push_str(&format!(
            "{},{},{:.10},{:.10},{}\n",
            score.frame_index,
            score.timestep,
            score.rms_abs_forces,
            score.rrmse_forces,
            score.selected
        ));
    }

    fs::write(path, text).map_err(|error| format!("Failed to write {}: {}", path.display(), error))
}

fn write_selected_xyz(
    path: &Path,
    frames: &[DumpFrame],
    selected_indices: &[usize],
    species: &[String],
) -> Result<(), String> {
    let mut text = String::new();

    for frame_index in selected_indices {
        let frame = frames
            .get(*frame_index)
            .ok_or_else(|| format!("Selected frame index {frame_index} is out of bounds"))?;
        let lattice = frame.bounds.lattice();

        text.push_str(&format!("{}\n", frame.atoms.len()));
        text.push_str(&format!(
            "Lattice=\"{:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10}\" Properties=species:S:1:pos:R:3 pbc=\"T T T\" timestep={}\n",
            lattice[0][0],
            lattice[0][1],
            lattice[0][2],
            lattice[1][0],
            lattice[1][1],
            lattice[1][2],
            lattice[2][0],
            lattice[2][1],
            lattice[2][2],
            frame.timestep
        ));

        for atom in &frame.atoms {
            let symbol = species_for_type(atom.atom_type, species)?;

            text.push_str(&format!(
                "{:<2} {:>16.8} {:>16.8} {:>16.8}\n",
                symbol, atom.position[0], atom.position[1], atom.position[2]
            ));
        }
    }

    fs::write(path, text).map_err(|error| format!("Failed to write {}: {}", path.display(), error))
}

fn read_lammps_species(input_lmp: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(input_lmp)
        .map_err(|error| format!("Failed to read {}: {}", input_lmp.display(), error))?;

    for line in text.lines() {
        let raw_line = line.trim();

        if let Some(mapping) = raw_line.strip_prefix("# alchemist_species") {
            let fields: Vec<&str> = mapping.split_whitespace().collect();

            if !fields.is_empty() {
                return fields
                    .iter()
                    .map(|field| symbol_from_token(field))
                    .collect();
            }
        }

        let line = line.split('#').next().unwrap_or("").trim();
        let fields: Vec<&str> = line.split_whitespace().collect();

        if fields.len() < 4 || fields[0] != "pair_coeff" || fields[1] != "*" || fields[2] != "*" {
            continue;
        }

        return fields[3..]
            .iter()
            .map(|field| symbol_from_token(field))
            .collect();
    }

    Err(format!(
        "Could not find a parseable '# alchemist_species ...' or 'pair_coeff * * ...' element mapping in {}",
        input_lmp.display()
    ))
}

fn fallback_species() -> Vec<String> {
    vec!["S".to_string(), "Cu".to_string()]
}

fn symbol_from_token(token: &str) -> Result<String, String> {
    if let Ok(atomic_number) = token.parse::<usize>() {
        return atomic_symbol(atomic_number)
            .map(str::to_string)
            .ok_or_else(|| format!("Unsupported atomic number in pair_coeff: {atomic_number}"));
    }

    Ok(token.to_string())
}

fn atomic_symbol(atomic_number: usize) -> Option<&'static str> {
    const SYMBOLS: [&str; 31] = [
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    ];

    SYMBOLS.get(atomic_number).copied()
}

fn species_for_type(atom_type: usize, species: &[String]) -> Result<&str, String> {
    species
        .get(atom_type.saturating_sub(1))
        .map(String::as_str)
        .ok_or_else(|| {
            format!(
                "LAMMPS atom type {} has no pair_coeff mapping ({})",
                atom_type,
                species.join(", ")
            )
        })
}

fn absolute_path_string(path: &Path) -> Result<String, String> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|error| format!("Failed to resolve {}: {}", path.display(), error))
            .map(|path| path.to_string_lossy().into_owned());
    }

    let parent = path
        .parent()
        .ok_or_else(|| format!("Path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("Path has no file name: {}", path.display()))?;
    let parent = parent
        .canonicalize()
        .map_err(|error| format!("Failed to resolve {}: {}", parent.display(), error))?;

    Ok(parent.join(file_name).to_string_lossy().into_owned())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();

    permissions.set_mode(0o755);

    fs::set_permissions(path, permissions)
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

fn expect_item<'a, I>(lines: &mut I, expected: &str) -> Result<(), String>
where
    I: Iterator<Item = &'a str>,
{
    let line = lines
        .next()
        .ok_or_else(|| format!("Expected {expected}, found end of dump"))?;

    if line.trim() == expected {
        Ok(())
    } else {
        Err(format!("Expected {expected}, found: {line}"))
    }
}

fn parse_next<'a, T, I>(lines: &mut I, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
    I: Iterator<Item = &'a str>,
{
    let line = lines
        .next()
        .ok_or_else(|| format!("Expected {label}, found end of dump"))?;

    line.trim()
        .parse::<T>()
        .map_err(|error| format!("Invalid {label} '{line}': {error}"))
}

fn column_index(columns: &[&str], name: &str) -> Result<usize, String> {
    columns
        .iter()
        .position(|column| *column == name)
        .ok_or_else(|| format!("LAMMPS dump atom table is missing '{name}' column"))
}

fn parse_field<T>(fields: &[&str], index: usize, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = fields
        .get(index)
        .ok_or_else(|| format!("Missing {label} field at column {index}"))?;

    value
        .parse::<T>()
        .map_err(|error| format!("Invalid {label} '{value}': {error}"))
}

fn parse_number<T>(value: &str, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("Invalid {label} '{value}': {error}"))
}
