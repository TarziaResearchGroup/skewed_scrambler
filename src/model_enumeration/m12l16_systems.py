"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import os
import pathlib
import time
import warnings
from collections import defaultdict

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"
import atomlite
import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import stk
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.utilities import eb_str, isomer_energy, multi_cmap
from model_enumeration.validation_utilities import (
    abead_d,
    analyse_cage,
    binder_bead,
    cbead_d,
    get_validation_forcefield,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

attempts = (
    # (None, None, None), # noqa: ERA001
    ("set", "kamada", 10),
    ("set", "spring", 10),
    # ("set", "spectral", 10), # noqa: ERA001
    # ("regraphed", "spring", 10),  # noqa: ERA001
    ("regraphed", "kamada", 10),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def make_opt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    stages = (
        "opt1",
        "smd",
        "shifted",
        "nx00",
        "nx10",
        "nx20",
        "nx30",
        "nx01",
        "nx11",
        "nx21",
        "nx31",
        "nx02",
        "nx12",
        "nx22",
        "nx32",
    )
    sources = {i: 0 for i in stages}
    # Produces low energy structures.
    lowe_sources = {i: 0 for i in stages}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        sources[entry.properties["source"]] += 1
        energy = entry.properties["energy_per_bb"]
        if energy < 1:
            lowe_sources[entry.properties["source"]] += 1

    ax.bar(
        stages,
        [lowe_sources[i] for i in stages],
        color="#086788",
        edgecolor="none",
        lw=2,
        label=f"{eb_str()} < 1.0",
    )
    ax.bar(
        stages,
        [sources[i] for i in stages],
        color="none",
        edgecolor="k",
        lw=2,
        label="all",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)  # , color=color)

    ax.legend(fontsize=16)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45)
    ax.set_xlabel("stage", fontsize=16)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_plot(  # noqa: C901, PLR0915
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    energies = defaultdict(list)
    bacs = defaultdict(list)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]

        if entry.properties["num_components"] > 1:
            continue

        energies[multi].append((bac_angle, energy, entry.key))
        bacs[bac_angle].append((multi, energy, entry.key))

    fig, (axx, ax) = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        height_ratios=[1, 4],
        sharex=True,
    )

    countsx = {}
    for i, multi in enumerate(sorted([int(i) for i in energies])):
        instable = False
        idx = str(multi)
        min_energy_tuple = min(energies[idx], key=lambda p: p[1])

        bac_line = []
        for bac_angle in sorted(bacs):
            rel_energies = [i[1] for i in energies[idx] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

            found_names_minus_mash = set()
            stable = []
            for entry_data in energies[idx]:
                if (
                    entry_data[0] == bac_angle
                    and entry_data[1] < isomer_energy()
                ):
                    names_minus_mash = "_".join(entry_data[-1].split("_")[:-1])

                    if names_minus_mash in found_names_minus_mash:
                        continue
                    found_names_minus_mash.add(names_minus_mash)
                    stable.append(entry_data)

            if len(stable) > 0:
                logging.info("stable cages: %s", stable)
                instable = True
            if bac_angle not in countsx:
                countsx[bac_angle] = 0
            countsx[bac_angle] += len(stable)

        ax.plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c=multi_cmap[str(multi)],
            ls="-",
            marker="o",
            markersize=10,
            markeredgecolor="k",
            alpha=1.0,
            zorder=2,
            label=(
                f"M{idx}: {min_energy_tuple[0]}, {min_energy_tuple[2]}"
                if instable
                else f"M{idx}"
            ),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_yscale("log")
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.set_ylabel(eb_str(), fontsize=16)

    axx.plot(
        list(countsx),
        [countsx[i] for i in countsx],
        c="k",
        marker="o",
        mec="k",
        zorder=2,
        markersize=6.0,
    )
    axx.tick_params(axis="both", which="major", labelsize=16)
    axx.set_ylabel("stable", fontsize=16)
    axx.set_ylim(0, None)

    leg = ax.legend(ncols=1, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.set_xlabel("target $bac$ angle [$^\\circ$]", fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_main_plot(
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    energies = defaultdict(list)
    bacs = defaultdict(list)
    graphs = defaultdict(set)

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]

        lig, multiplier, idx, mash_idx = entry.key.split("_")
        if entry.properties["num_components"] > 1:
            continue

        energies[multi].append((bac_angle, energy, entry.key))
        bacs[bac_angle].append((multi, energy, entry.key))
        graphs[multi].add(idx)

    fig, ax = plt.subplots(figsize=(8, 2))
    for i, multi in enumerate(sorted([int(i) for i in energies])):
        idx = str(multi)
        logging.info(
            "there are %s graphs in m=%s", len(list(graphs[idx])), idx
        )

        bac_line = []
        for bac_angle in sorted(bacs):
            rel_energies = [i[1] for i in energies[idx] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

            found_names_minus_mash = set()
            stable = []
            for entry_data in energies[idx]:
                if (
                    entry_data[0] == bac_angle
                    and entry_data[1] < isomer_energy()
                ):
                    names_minus_mash = "_".join(entry_data[-1].split("_")[:-1])

                    if names_minus_mash in found_names_minus_mash:
                        continue
                    found_names_minus_mash.add(names_minus_mash)
                    stable.append(entry_data)

        ax.plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c=multi_cmap[str(multi)],
            ls="-",
            marker="o",
            markersize=10,
            markeredgecolor="k",
            alpha=1.0,
            zorder=2,
            label=f"M{idx}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_xlim(85, 155)
    ax.set_xticks([90, 100, 110, 120, 130, 140, 150])
    ax.set_ylim(0, 2)

    leg = ax.legend(ncols=3, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.set_xlabel("$bac$ angle [$^\\circ$]", fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_timings_plot(
    figure_dir: pathlib.Path,
    timing_file: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    timings = pl.read_csv(
        timing_file,
        has_header=False,
        new_columns=[
            "lig",
            "multi",
            "num_vertices",
            "idx",
            "mash_idx",
            "opt-time/s",
            "ana-time/s",
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        timings["num_vertices"],
        timings["opt-time/s"],
        c="tab:blue",
        s=100,
        alpha=1,
        label="optimisation time",
        ec="k",
        rasterized=True,
    )
    ax.scatter(
        timings["num_vertices"],
        timings["ana-time/s"],
        c="tab:orange",
        s=100,
        alpha=1,
        label="analysis time",
        ec="k",
        rasterized=True,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num. vertices", fontsize=16)
    ax.set_ylabel("time [s]", fontsize=16)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_summary_plot(
    database_path: pathlib.Path,
    structure_dir: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    energies = defaultdict(list)

    tops = figure_dir / "ivalidation_tops.txt"
    if tops.exists():
        tops.unlink()

    to_save = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        bite_angle = (bac_angle - 90) * 2

        vstr = entry.key.split("_")[2]

        if entry.properties["num_components"] > 1:
            continue
        energies[(multi, bite_angle)].append(
            (round(energy, 4), vstr, entry.key)
        )
        if energy < isomer_energy():
            to_save.append(entry.key)
            rdkit_molecule = atomlite.json_to_rdkit(entry.molecule)
            stk.BuildingBlock.init_from_rdkit_mol(rdkit_molecule).write(
                structure_dir / f"{entry.key}_optc.mol"
            )

        if multi == "4":
            rdkit_molecule = atomlite.json_to_rdkit(entry.molecule)
            stk.BuildingBlock.init_from_rdkit_mol(rdkit_molecule).write(
                structure_dir / f"{entry.key}_optc.mol"
            )

    with tops.open("w") as f:
        for ts in sorted(to_save):
            file_ = structure_dir / f"{ts}_optc.mol"
            if file_.exists():
                f.write(f"{file_} ")

    vmin = 0
    vmax = 1
    for multi, bite_angle in energies:
        sorted_energies = sorted(
            energies[(multi, bite_angle)], key=lambda p: p[0]
        )
        min_energy = sorted_energies[0]

        cgx.utilities.AtomliteDatabase(database_path).get_molecule(
            min_energy[2]
        ).with_centroid((0, 0, 0)).write(
            structure_dir / f"{min_energy[2]}_optc.mol"
        )

        x = int(bite_angle)
        y = int(multi)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap="Blues_r",
        )
        ax.text(
            x=x,
            y=y,
            s=min_energy[1],
            horizontalalignment="center",
            verticalalignment="center_baseline",
            color="w" if min_energy[0] < 0.5 else "k",  # noqa: PLR2004
            fontsize=6,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("target bite angle [deg]", fontsize=16)
    ax.set_ylabel("multiplier", fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(eb_str(), fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/tarziaa/workingspace/skewed_production/")
    (wd / "figures").mkdir(exist_ok=True)
    run_prefix = "m12l16"
    figure_dir = wd / "figures" / run_prefix

    calculation_dir = wd / f"{run_prefix}_calculations"
    structure_dir = wd / f"{run_prefix}_structures"
    ligand_dir = wd / f"{run_prefix}_ligands"
    data_dir = wd / f"{run_prefix}_data"
    database_path = data_dir / f"{run_prefix}_run.db"

    calculation_dir.mkdir(exist_ok=True)
    structure_dir.mkdir(exist_ok=True)
    ligand_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    constant_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        "nd": ("bond", 2.85, 1e5),
        "bd": ("bond", 1.1, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "dnd": ("angle", 120, 1e2),
        "mbd": ("angle", 180, 1e2),
        "ndb": ("angle", 180, 1e2),
        # Torsions.
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "n": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
    }

    # Define beads.
    bead_library = cgx.molecular.BeadLibrary.from_bead_types(
        # Type and coordination.
        {"m": 4, "n": 3, "b": 2, "d": 2}
    )

    tritopi_prec = cgx.molecular.ThreeC1Arm(
        bead=bead_library.get_from_type("n"),
        abead1=bead_library.get_from_type("d"),
    )
    tetrato_prec = cgx.molecular.FourC1Arm(
        bead=bead_library.get_from_type("m"),
        abead1=bead_library.get_from_type("b"),
    )
    stoichimetry_l_m = (2, 1)
    studies = ({"ndb": 160, "multi": (1, 2, 3, 4)},)
    ligands = {
        str(i["bac"]): {
            "forcefield": cgx.systems_optimisation.get_forcefield_from_dict(
                identifier=str(i["bac"]),
                prefix="min_val",
                vdw_bond_cutoff=2,
                present_beads=bead_library.get_present_beads(),
                definer_dict=cgx.systems_optimisation.merge_definer_dicts(
                    original_definer_dict=constant_definer_dict,
                    new_definer_dicts=[{"ndb": ("angle", i["ndb"], 1e2)}],
                ),
            ),
            "tritopic": tritopi_prec,
            "tetra": tetrato_prec,
            "multipliers": i["multi"],
        }
        for i in studies
    }

    if args.run:
        for lig in ligands:
            forcefield = ligands[lig]["forcefield"]
            tritopic = ligands[lig]["tritopic"]
            tetra = ligands[lig]["tetra"]

            tritopic_name = (
                f"{tritopic.get_name()}_f{forcefield.get_identifier()}"
            )
            ditopic_bb = cgx.utilities.optimise_ligand(
                molecule=tritopic.get_building_block(),
                name=tritopic_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            ditopic_bb.write(str(ligand_dir / f"{tritopic_name}_optl.mol"))
            ditopic_bb = ditopic_bb.clone()

            tetra_name = f"{tetra.get_name()}_f{forcefield.get_identifier()}"
            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            tetra_bb.write(str(ligand_dir / f"{tetra_name}_optl.mol"))
            tetra_bb = tetra_bb.clone()

            for multiplier in ligands[lig]["multipliers"]:
                iterator = cgx.scram.SkewedTopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichimetry_l_m[1] * multiplier,
                        tritopic_bb: stoichimetry_l_m[0] * multiplier,
                    },
                )

                logging.info(
                    "doing: ligand %s, multi %s with %s graphs",
                    lig,
                    multiplier,
                    iterator.count_graphs(),
                )
                raise SystemExit

                for topology_code in iterator.yield_graphs():
                    for midx, (method, layout_type, scale) in enumerate(
                        attempts
                    ):
                        name = f"{lig}_{multiplier}_{topology_code.idx}_{midx}"

                        try:
                            if method == "regraphed":
                                constructed_molecule = (
                                    cgx.scram.get_regraphed_molecule(
                                        layout_type=layout_type,  # type:ignore[arg-type]
                                        scale=scale,  # type:ignore[arg-type]
                                        topology_code=topology_code,
                                        iterator=iterator,
                                        configuration=None,
                                    )
                                )

                            elif method == "set":
                                constructed_molecule = (
                                    cgx.scram.get_vertexset_molecule(
                                        layout_type=layout_type,  # type:ignore[arg-type]
                                        scale=scale,  # type:ignore[arg-type]
                                        topology_code=topology_code,
                                        iterator=iterator,
                                        configuration=None,
                                    )
                                )

                            else:
                                constructed_molecule = (
                                    cgx.scram.try_except_construction(
                                        iterator=iterator,
                                        topology_code=topology_code,
                                        configuration=None,
                                        vertex_positions=None,
                                    ).with_centroid(np.array((0.0, 0.0, 0.0)))
                                )

                        except ValueError:
                            continue

                        # Optimise and save.
                        logger.info("building %s", name)

                        try:
                            raise SystemExit("implement this cheap one")
                            conformer = custom_optimise_cage(
                                molecule=constructed_molecule,
                                name=name,
                                output_dir=calculation_dir,
                                forcefield=forcefield,
                                platform=None,
                                database_path=conformer_db_path,
                                potential_names=[],
                            )

                            if (
                                conformer is not None
                                and conformer.energy_decomposition[
                                    "total energy"
                                ][0]
                                is not None
                            ):
                                num_components = len(
                                    stko.Network.init_from_molecule(
                                        conformer.molecule
                                    ).get_connected_components()
                                )

                                energy_per_bb = (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                )

                                properties = {
                                    "forcefield_dict": (
                                        forcefield.get_forcefield_dictionary()
                                    ),
                                    "energy_per_bb": energy_per_bb,
                                    "num_components": num_components,
                                    "mash_idx": midx,
                                    "config_name": eval_name,
                                    "num_bbs": (
                                        iterator.get_num_building_blocks()
                                    ),
                                    "topology_code_vmap": tuple(
                                        (int(i[0]), int(i[1]))
                                        for i in topology_code.vertex_map
                                    ),
                                }
                                cgx.utilities.AtomliteDatabase(
                                    conformer_db_path
                                ).add_properties(
                                    key=name,
                                    property_dict=properties,  # type:ignore[arg-type]
                                )

                        except (openmm.OpenMMException, ValueError):
                            logger.info("failed optimisation of %s", name)

                    conformer_db = cgx.utilities.AtomliteDatabase(
                        conformer_db_path
                    )
                    min_energy_structure = None
                    min_energy = float("inf")
                    min_energy_key = None
                    for entry in conformer_db.get_entries():
                        # Handle failed opt.
                        if "energy_per_bb" not in entry.properties:
                            continue

                        if (
                            float(entry.properties["energy_per_bb"])  # type:ignore[arg-type]
                            < min_energy
                        ):
                            min_energy = float(
                                entry.properties["energy_per_bb"]  # type:ignore[arg-type]
                            )
                            min_energy_structure = conformer_db.get_molecule(
                                key=entry.key
                            )
                            min_energy_key = entry.key

                    if min_energy_structure is None:
                        return min_energy

                    # To file.
                    min_energy_structure.write(  # type:ignore[union-attr]
                        str(struct_output / f"{eval_name}_optc.mol")
                    )
                    logger.info(
                        "Time taken for full opt: %s",
                        round(time.time() - s, 2),
                    )

                    # To database.
                    cgx.utilities.AtomliteDatabase(database_path).add_molecule(
                        molecule=min_energy_structure,  # type:ignore[arg-type]
                        key=eval_name,
                    )
                    initial_properties = (
                        cgx.utilities.AtomliteDatabase(conformer_db_path)
                        .get_entry(min_energy_key)
                        .properties
                    )
                    cgx.utilities.AtomliteDatabase(
                        database_path
                    ).add_properties(
                        key=eval_name,
                        property_dict=initial_properties,
                    )

                    nauty_cert = get_nauty_certificate(graph)
                    properties = {
                        "grpsize": get_nauty_automoprhism_grpsize(graph),
                        "nauty_key": str(nauty_cert),
                        "lowest_e_of_mash": True,
                        "skewness_upper_bound": skewness_upper_bound(graph),
                    }
                    cgx.utilities.AtomliteDatabase(
                        database_path
                    ).add_properties(
                        key=eval_name,
                        property_dict=properties,
                    )

                    # To database.
                    cgx.utilities.AtomliteDatabase(
                        graph_database_path
                    ).add_molecule(
                        molecule=min_energy_structure,  # type:ignore[arg-type]
                        key=str(nauty_cert),
                    )
                    properties = {
                        "eval_name": eval_name,
                    }
                    cgx.utilities.AtomliteDatabase(
                        graph_database_path
                    ).add_properties(
                        key=str(nauty_cert),
                        property_dict=properties,
                    )
                make_plot(
                    database_path=database_path,
                    figure_dir=figure_dir,
                    filename=f"{run_prefix}_1.png",
                )

    make_summary_plot(
        database_path=database_path,
        structure_dir=structure_dir,
        figure_dir=figure_dir,
        filename=f"{run_prefix}_2.png",
    )
    make_main_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename=f"{run_prefix}_1c.png",
    )
    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename=f"{run_prefix}_1.png",
    )
    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename=f"{run_prefix}_5.png",
    )


if __name__ == "__main__":
    main()
