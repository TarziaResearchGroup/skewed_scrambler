"""Script to generate and optimise CG models."""

import argparse
import logging
import os
import pathlib
import time
from collections import abc, defaultdict

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"
import agx
import atomlite
import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    a2bead_d,
    abead_c,
    abead_d,
    binder_bead,
    c2bead_d,
    cbead_c,
    cbead_d,
    construct_cage,
    custom_optimise_cage,
    e2bead_d,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)
from model_enumeration.utilities import (
    eb_str,
    isomer_energy,
    multi_cmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
attempts = (
    # (None, None, None), # noqa: ERA001
    ("set", "kamada", 10),
    ("set", "spring", 10),
    # ("set", "spectral", 10), # noqa: ERA001
    # ("regraphed", "spring", 10),  # noqa: ERA001
    ("regraphed", "kamada", 10),
)
seed_cs = {
    4: "tab:blue",
    12689: "tab:orange",
    18: "tab:green",
    999: "tab:red",
    142: "tab:purple",
    6582: "tab:cyan",
}


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
    width_height: tuple[float, float] = (7, 10),
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=width_height)
    energies = {}

    xs = []

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        if multi not in xs:
            xs.append(multi)

        pair = tuple(entry.properties["pair"].split("_"))
        if len(pair) > 3:  # noqa: PLR2004
            msg = f"is {pair} right? ({entry.properties['pair']})"
            raise RuntimeError(msg)
        if len(pair) == 3:  # noqa: PLR2004
            pair = (pair[0], pair[1] + "_" + pair[2])

        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]
        midx = entry.properties["mash_idx"]
        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in energies:
            energies[(pair, multi)] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[(pair, multi)].append((round(energy, 4), tidx, bidx, midx))

    # create the new map
    cmap = plt.cm.Blues_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )

    # define the bins and normalize
    bounds = [0, 0.3, 1.0, 5.0, 10.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    for (pair, multi), evalues in energies.items():
        sorted_energies = sorted(evalues, key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = [i[0] for i in pairs].index(pair)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap=cmap,
            norm=norm,
        )
        ax.text(
            x=x + 0.5,
            y=y,
            s=f"t:{min_energy[1]},b:{min_energy[2]}",
            horizontalalignment="center",
            verticalalignment="center_baseline",
            color="k",
            fontsize=10,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(pairs))))
    ax.set_yticklabels(["_".join(i) for i in [i[0] for i in pairs]])

    for i in list(range(len(xs))):
        ax.axvline(int(i) + 0.8, c="k", alpha=0.5)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"1:1:1 {eb_str()}", fontsize=16)

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


def make_summary_plot2(  # noqa: C901, PLR0912
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    structure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
) -> dict:
    """Visualise energies."""
    fig, (axx, ax) = plt.subplots(
        nrows=2,
        figsize=(16, 6),
        height_ratios=[1, 3],
        sharex=True,
    )

    x_multi_mins = {i: defaultdict(float) for i in multi_cmap}
    x_count = {i: defaultdict(int) for i in multi_cmap}
    min_at_all_xs = defaultdict(int)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])

        pair = tuple(entry.properties["pair"].split("_"))
        if len(pair) > 3:  # noqa: PLR2004
            msg = f"is {pair} right? ({entry.properties['pair']})"
            raise RuntimeError(msg)
        if len(pair) == 3:  # noqa: PLR2004
            pair = (pair[0], pair[1] + "_" + pair[2])

        x = [i[0] for i in pairs].index(pair)
        x_count[multi][x] += 1
        energy = entry.properties["energy_per_bb"]

        if energy < 1:
            stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(entry.molecule)
            ).write(structure_dir / f"{entry.key}_optc.mol")

        if entry.properties["num_components"] > 1:
            continue

        if x not in x_multi_mins[multi]:
            x_multi_mins[multi][x] = energy
        else:
            x_multi_mins[multi][x] = min((x_multi_mins[multi][x], energy))

        if x not in min_at_all_xs:
            min_at_all_xs[x] = energy
        else:
            min_at_all_xs[x] = min((min_at_all_xs[x], energy))

    for i in range(len(pairs) - 1):
        ax.axvline(x=i + 0.5, c="k", alpha=0.2)
        axx.axvline(x=i + 0.5, c="k", alpha=0.2)

    for multi in multi_cmap:
        if len(x_multi_mins[multi]) == 0:
            continue
        edict = x_multi_mins[multi]

        ax.plot(
            sorted(edict),
            [edict[i] for i in sorted(edict)],
            c="none",
            markerfacecolor=multi_cmap[multi],
            mec="k",
            marker="o",
            alpha=1,
            markersize=12,
        )
        axx.plot(
            list(x_count[multi]),
            [x_count[multi][i] for i in x_count[multi]],
            c="none",
            markerfacecolor=multi_cmap[multi],
            mec="k",
            marker="o",
            zorder=2,
            markersize=12,
        )

    ax.plot(
        sorted(min_at_all_xs),
        [min_at_all_xs[i] for i in sorted(min_at_all_xs)],
        c="k",
        alpha=1,
        zorder=-1,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(pairs))))
    ax.set_xticklabels(
        ["_".join(i) for i in [i[0] for i in pairs]], rotation=90
    )
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)

    axx.tick_params(axis="both", which="major", labelsize=16)
    axx.set_ylabel("calcs", fontsize=16)

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


def make_opt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    stages = (
        "opt1",
        "nx00",
        "nx10",
        "nx01",
        "nx11",
        "nx02",
        "nx12",
        "ropt",
    )

    sources = {i: 0 for i in stages}
    mide_sources = {i: 0 for i in stages}
    lowe_sources = {i: 0 for i in stages}

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue
        if entry.properties["stoichstring"] != "9-9-9":
            continue

        sources[entry.properties["source"]] += 1

        energy = entry.properties["energy_per_bb"]
        if energy < 10:  # noqa: PLR2004
            mide_sources[entry.properties["source"]] += 1
        if energy < 1:
            lowe_sources[entry.properties["source"]] += 1

    ax.bar(
        stages,
        [lowe_sources[i] for i in stages],
        color="tab:blue",
        edgecolor="none",
        label=f"{eb_str()} < 1.0",
        zorder=1,
    )
    ax.bar(
        stages,
        [mide_sources[i] for i in stages],
        color="tab:gray",
        edgecolor="none",
        lw=2,
        label=f"{eb_str()} < 10.0",
        alpha=0.5,
        zorder=0,
    )
    ax.bar(
        stages,
        [sources[i] for i in stages],
        color="none",
        edgecolor="k",
        lw=2,
        label="all",
        zorder=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)  # , color=color)
    ax.legend(fontsize=16)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45)
    ax.set_xlabel("stage", fontsize=16)
    ax.set_yscale("log")

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


def add_generation_information(
    database_path: pathlib.Path,
    generation: cgx.systems_optimisation.Generation,
    seed: int,
    generation_id: int,
) -> None:
    """Add generation information to the database."""
    for cs in generation.chromosomes:
        topology_idx, topology_code = cs.get_topology_information()
        building_block_config = cs.get_building_block_configurations()[0]
        name = f"{cs.prefix}_{topology_idx}_b{building_block_config.idx}"
        entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(name)
        if "generation_id" not in entry.properties:
            cgx.utilities.AtomliteDatabase(database_path).add_properties(
                key=name,
                property_dict={
                    "generation_id": generation_id,
                    "generation_seed": seed,
                },
            )


def fitness_function(  # noqa: PLR0913
    chromosome: cgx.systems_optimisation.Chromosome,
    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator,  # noqa: ARG001
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,  # noqa: ARG001
    structure_output: pathlib.Path,  # noqa: ARG001
    options: dict,
) -> float:
    """Compute fitness."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    topology_idx, topology_code = chromosome.get_topology_information()
    building_block_config = chromosome.get_building_block_configurations()[0]

    name = f"{chromosome.prefix}_{topology_idx}_b{building_block_config.idx}"

    entry = database.get_entry(name)

    if entry.properties["is_duplicate"]:
        energy = database.get_entry(
            entry.properties["duplicate_of"]
        ).properties["energy_per_bb"]

    else:
        energy = entry.properties["energy_per_bb"]
    fitness = np.exp(-energy * options["beta"])
    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


def structure_function(  # noqa: C901, PLR0912, PLR0915
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,
) -> None:
    """Compute structure."""
    database = cgx.utilities.AtomliteDatabase(database_path)

    topology_idx, topology_code = chromosome.get_topology_information()
    building_block_config = chromosome.get_building_block_configurations()[0]

    base_name = (
        f"{chromosome.prefix}_{topology_idx}_b{building_block_config.idx}"
    )
    l1, l2, stoichstring = chromosome.prefix.split("_")
    multiplier = stoichstring.split("-")[2]

    if database.has_molecule(base_name):
        return

    # Check if this has been run before.
    known_entry = None
    for entry in database.get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        try:
            entry_tc = options["topology_codes"][
                entry.properties["topology_idx"]
            ][1]
            entry_bb_config = options["bb_configs"][
                entry.properties["bb_config_idx"]
            ]
        except (KeyError, IndexError):
            continue

        known_stoichstring = entry.properties["stoichstring"]
        known_pair = entry.properties["pair"]
        if f"{l1}_{l2}" != known_pair:
            continue
        if stoichstring != known_stoichstring:
            continue

        # Testing bb-config aware graph check.
        configured = agx.ConfiguredCode(topology_code, building_block_config)
        if not agx.utilities.is_configured_code_isomorphic(
            test_code=configured,
            run_topology_codes=[agx.ConfiguredCode(entry_tc, entry_bb_config)],
        ):
            known_entry = entry
            break

    # Try to avoid recalculation if possible.
    if (
        known_entry is not None
        and known_entry.properties["config_name"] != base_name
    ):
        database.add_molecule(
            key=base_name,
            molecule=database.get_molecule(known_entry.key),
        )
        database.add_properties(
            key=base_name,
            property_dict={
                "is_duplicate": True,
                "duplicate_of": known_entry.key,
                "opt_passed": True,
            },
        )

        try:
            nd_ = known_entry.properties["num_duplicates"] + 1
        except KeyError:
            nd_ = 1
        database.add_properties(
            key=known_entry.key,
            property_dict={"num_duplicates": nd_},
        )

        logging.info("%s is duplicate", base_name)
        return

    # Iterate over mashes.
    conformer_db_path = calculation_output / f"{base_name}.db"
    for midx, (method, layout_type, scale) in enumerate(attempts):
        name = f"{base_name}_{midx}"

        try:
            constructed_molecule = construct_cage(
                method=method,
                layout_type=layout_type,  # type:ignore[arg-type]
                scale=scale,  # type:ignore[arg-type]
                topology_code=topology_code,
                iterator=options["iterator"],
                configuration=None,
                vertex_positions=None,
            )
        except ValueError:
            continue

        # Optimise and save.
        logging.info("building %s", name)

        try:
            conformer = custom_optimise_cage(
                molecule=constructed_molecule,
                name=name,
                output_dir=calculation_output,
                forcefield=options["forcefield"],
                platform=None,
                database_path=conformer_db_path,
                potential_names=[],
            )

            if (
                conformer is not None
                and conformer.energy_decomposition["total energy"][0]
                is not None
            ):
                num_components = len(
                    stko.Network.init_from_molecule(
                        conformer.molecule
                    ).get_connected_components()
                )

                energy_per_bb = cgx.utilities.get_energy_per_bb(
                    energy_decomposition=(conformer.energy_decomposition),
                    number_building_blocks=(
                        options["iterator"].get_num_building_blocks()
                    ),
                )

                properties = {
                    "forcefield_dict": (
                        options["forcefield"].get_forcefield_dictionary()
                    ),
                    "energy_per_bb": energy_per_bb,
                    "num_components": num_components,
                    "mash_idx": midx,
                    "l1": l1,
                    "l2": l2,
                    "pair": f"{l1}_{l2}",
                    "stoichstring": stoichstring,
                    "config_name": base_name,
                    "topology_idx": topology_idx,
                    "bb_config_idx": building_block_config.idx,
                    "num_bbs": (options["iterator"].get_num_building_blocks()),
                    "multiplier": str(multiplier),
                    "topology_code_vmap": tuple(
                        (int(i[0]), int(i[1]))
                        for i in topology_code.vertex_map
                    ),
                    # Add here, if it gets here, then it is not duplicate.
                    "is_duplicate": False,
                    "num_duplicates": 0,
                }
                cgx.utilities.AtomliteDatabase(
                    conformer_db_path
                ).add_properties(
                    key=name,
                    property_dict=properties,  # type:ignore[arg-type]
                )

        except (OpenMMException, ValueError):
            logging.info("failed optimisation of %s", name)

    conformer_db = cgx.utilities.AtomliteDatabase(conformer_db_path)
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
            min_energy_structure = conformer_db.get_molecule(key=entry.key)
            min_energy_key = entry.key

    # To file.
    min_energy_structure.write(  # type:ignore[union-attr]
        str(structure_output / f"{base_name}_optc.mol")
    )

    # To database.
    cgx.utilities.AtomliteDatabase(database_path).add_molecule(
        molecule=min_energy_structure,  # type:ignore[arg-type]
        key=base_name,
    )
    initial_properties = (
        cgx.utilities.AtomliteDatabase(conformer_db_path)
        .get_entry(min_energy_key)
        .properties
    )
    cgx.utilities.AtomliteDatabase(database_path).add_properties(
        key=base_name,
        property_dict=initial_properties,
    )

    properties = {"lowest_e_of_mash": True, "is_base": True}
    cgx.utilities.AtomliteDatabase(database_path).add_properties(
        key=base_name, property_dict=properties
    )
    return


def plot_timings(figure_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    """Plot timings."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for num_processes in (1, 2, 3, 4, 5, 6, 7, 8):
        timing_file = data_dir / f"np_{num_processes}.txt"
        if not timing_file.exists():
            continue
        with timing_file.open("r") as f:
            lines = f.readlines()
        str_times = [float(i.strip().split(",")[0]) for i in lines]
        fit_times = [float(i.strip().split(",")[1]) for i in lines]
        if num_processes == 1:
            lbl1 = "structure"
            lbl2 = "fitness"
        else:
            lbl1 = None
            lbl2 = None
        ax.scatter(
            [num_processes for i in str_times],
            str_times,
            c="tab:blue",
            s=40,
            edgecolor="none",
            alpha=0.4,
        )
        ax.scatter(
            [num_processes for i in fit_times],
            fit_times,
            c="tab:orange",
            s=40,
            edgecolor="none",
            alpha=0.4,
        )
        ax.scatter(
            num_processes,
            sum(str_times) / len(str_times),
            c="tab:blue",
            s=100,
            edgecolor="k",
            label=lbl1,
        )
        ax.scatter(
            num_processes,
            sum(fit_times) / len(fit_times),
            c="tab:orange",
            s=100,
            edgecolor="k",
            label=lbl2,
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num processes", fontsize=16)
    ax.set_ylabel("time [s]", fontsize=16)
    ax.set_ylim(0, None)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "timings.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_counters(  # noqa: C901, PLR0912, PLR0915
    database_path: pathlib.Path,
    stoichstring: str,
    figure_dir: pathlib.Path,
    filename: str,
    chosen_pair: str,
) -> dict:
    """Visualise energies."""
    fig, (axx, ax1) = plt.subplots(ncols=2, figsize=(16, 5))

    count_has_parallels = {True: 0, False: 0}
    has_parallels = {True: [], False: []}
    gen_entries = {}
    done = 0
    done_path = {}
    min_gen_energy = {}
    maxbbidx = 0
    maxtidx = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair != chosen_pair:
            continue
        if entry.properties["stoichstring"] != stoichstring:
            continue

        energy = entry.properties["energy_per_bb"]
        count_has_parallels[entry.properties["contains_parallels"]] += 1
        has_parallels[entry.properties["contains_parallels"]].append(energy)

        if energy < 0.1:  # noqa: PLR2004
            c = "tab:blue"
            zorder = 2
            s = 50
        elif energy < 1:
            c = "tab:orange"
            zorder = 1
            s = 20
        else:
            c = "tab:gray"
            zorder = 0
            s = 10

        if "generation_id" in entry.properties:
            if entry.properties["generation_id"] not in gen_entries:
                gen_entries[entry.properties["generation_id"]] = []
                min_gen_energy[entry.properties["generation_id"]] = float(
                    "inf"
                )
            gen_entries[entry.properties["generation_id"]].append(entry)
            min_gen_energy[entry.properties["generation_id"]] = min(
                (energy, min_gen_energy[entry.properties["generation_id"]])
            )

        ax1.scatter(
            entry.properties["topology_idx"],
            entry.properties["bb_config_idx"],
            c=c,
            s=s,
            alpha=1,
            zorder=zorder,
        )
        done += 1
        maxbbidx = max((maxbbidx, entry.properties["bb_config_idx"]))
        maxtidx = max((maxtidx, entry.properties["topology_idx"]))
        if "generation_id" in entry.properties:
            if entry.properties["generation_id"] not in done_path:
                done_path[entry.properties["generation_id"]] = []

            if energy == min_gen_energy[entry.properties["generation_id"]]:
                done_path[entry.properties["generation_id"]] = [
                    entry.properties["topology_idx"],
                    entry.properties["bb_config_idx"],
                ]

    if len(done_path) > 0:
        ax1.scatter(done_path[0][0], done_path[0][1], c="r", s=50)
        ax1.plot(
            [i[0] for i in done_path.values()],
            [i[1] for i in done_path.values()],
            c="k",
            lw=1,
            zorder=-1,
        )
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("tidx", fontsize=16)
    ax1.set_ylabel("bidx", fontsize=16)
    ax1.set_title(f"{done} of {maxbbidx * maxtidx}", fontsize=16)

    nadded = []
    nduplicates = []
    nparallels = []
    cumul = []
    for gen_id, gen_list in gen_entries.items():
        nduplicates.append(
            sum([1 if i.properties["is_duplicate"] else 0 for i in gen_list])
        )
        nparallels.append(
            sum(
                [
                    1 if i.properties["contains_parallels"] else 0
                    for i in gen_list
                ]
            )
        )
        nadded.append(len(gen_list))
        if gen_id == 0:
            cumul.append(len(gen_list))
        else:
            try:
                cumul.append(cumul[-1] + len(gen_list))
            except IndexError:
                break

    axx.plot(
        nparallels,
        markerfacecolor="#F9A03F",
        lw=2,
        label="n. parallels",
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )
    axx.plot(
        nadded,
        markerfacecolor="#086788",
        lw=2,
        label="n. added",
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )
    axx.plot(
        nduplicates,
        markerfacecolor="#7A8B99",
        lw=2,
        label="n. duplicates",
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )

    axx.tick_params(axis="both", which="major", labelsize=16)
    axx.set_xlabel("generation", fontsize=16)
    axx.set_ylabel("number", fontsize=16)
    axx.legend(fontsize=16)
    axx.set_ylim(0, None)
    if len(cumul) > 0:
        axx.set_title(f"tot. gen: {cumul[-1]}", fontsize=16)

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


def plot_vs_parallels(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    stoichstring: str,
    filename: str,
    chosen_pair: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    count_has_parallels = {True: 0, False: 0}
    has_parallels = {True: [], False: []}
    present_tidx = defaultdict(list)
    present_bidx = defaultdict(list)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair != chosen_pair:
            continue
        if entry.properties["stoichstring"] != stoichstring:
            continue

        energy = entry.properties["energy_per_bb"]
        count_has_parallels[entry.properties["contains_parallels"]] += 1
        has_parallels[entry.properties["contains_parallels"]].append(energy)
        present_tidx[entry.properties["topology_idx"]].append(energy)
        present_bidx[entry.properties["bb_config_idx"]].append(energy)

    steps = range(len(count_has_parallels) + 6 - 1, -1, -1)
    xmin = 0
    xmax = max(has_parallels[False])
    xwidth = 0.2
    ystep = 1
    xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)

    for i, value in enumerate(has_parallels):
        ax.hist(
            x=has_parallels[value],
            bins=xbins,
            density=True,
            bottom=steps[i] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            label=f"parallels: {value}",
        )

    counted = {i: len(present_tidx[i]) for i in present_tidx}
    top_3 = sorted(counted, key=counted.get, reverse=True)[:3]

    for i, tidx in enumerate(top_3):
        j = i + len(has_parallels)
        ax.hist(
            x=present_tidx[tidx],
            bins=xbins,
            density=True,
            bottom=steps[j] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            label=f"tidx: {tidx}",
        )

    counted = {i: len(present_bidx[i]) for i in present_bidx}
    top_3 = sorted(counted, key=counted.get, reverse=True)[:3]

    for i, bidx in enumerate(top_3):
        j = i + len(has_parallels) + 3
        ax.hist(
            x=present_bidx[bidx],
            bins=xbins,
            density=True,
            bottom=steps[j] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            label=f"bidx: {bidx}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(eb_str(), fontsize=16)

    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.set_ylim(0, (steps[0] + 1.5) * ystep)
    ax.legend(fontsize=16)
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


def plot_elite_graphs(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    stoichstring: str,
    filename: str,
    chosen_pair: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    present_tidx = defaultdict(list)
    all_energies = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair != chosen_pair:
            continue

        if entry.properties["stoichstring"] != stoichstring:
            continue

        energy = entry.properties["energy_per_bb"]
        all_energies.append(energy)
        present_tidx[entry.properties["topology_idx"]].append(energy)

    steps = range(4 - 1, -1, -1)
    counted = {i: len(present_tidx[i]) for i in present_tidx}
    top_3 = sorted(counted, key=counted.get, reverse=True)[:3]

    xmin = 0
    xmax = max(all_energies)
    xwidth = 0.1
    ystep = 1
    xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
    ax.hist(
        x=all_energies,
        bins=xbins,
        density=True,
        bottom=steps[0] * ystep,
        histtype="stepfilled",
        stacked=True,
        linewidth=1.0,
        edgecolor="k",
        label="all",
    )
    ax.plot(
        (min(all_energies), min(all_energies)),
        (steps[0] * ystep, ystep + (steps[0] * ystep)),
        zorder=-2,
        c="k",
    )

    for i, tidx in enumerate(top_3):
        ax.hist(
            x=present_tidx[tidx],
            bins=xbins,
            density=True,
            bottom=steps[i + 1] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            label=f"tidx: {tidx}",
        )

        ax.plot(
            (min(present_tidx[tidx]), min(present_tidx[tidx])),
            (steps[i + 1] * ystep, steps[i] * ystep),
            zorder=-2,
            c="k",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(eb_str(), fontsize=16)
    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.set_ylim(0, (steps[0] + 1.5) * ystep)
    ax.legend(fontsize=16)
    ax.set_xscale("log")

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


def plot_idx_completion(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    stoichstring: str,
    filename: str,
    chosen_pair: str,
) -> dict:
    """Visualise energies."""
    fig, (ax, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(16, 5))

    present_tidx = defaultdict(int)
    present_bidx = defaultdict(int)
    max_tidx = 0
    max_bidx = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair != chosen_pair:
            continue
        if entry.properties["stoichstring"] != stoichstring:
            continue

        present_tidx[entry.properties["topology_idx"]] += 1
        present_bidx[entry.properties["bb_config_idx"]] += 1
        max_tidx = max((max_tidx, entry.properties["topology_idx"]))
        max_bidx = max((max_bidx, entry.properties["bb_config_idx"]))

    top_10 = sorted(present_tidx, key=present_tidx.get, reverse=True)[:10]
    ax.bar(
        x=range(len(top_10)),
        height=[present_tidx[i] for i in top_10],
        facecolor="tab:blue",
    )
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(list(top_10))
    top_10 = sorted(present_bidx, key=present_bidx.get, reverse=True)[:10]
    ax1.bar(
        x=range(len(top_10)),
        height=[present_bidx[i] for i in top_10],
        facecolor="tab:orange",
    )
    ax1.set_xticks(range(len(top_10)))
    ax1.set_xticklabels(list(top_10))
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("tidx", fontsize=16)
    ax.set_ylabel("count", fontsize=16)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("bidx", fontsize=16)

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


def plot_energies(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    seeds = {4: 0, 12689: 10, 18: 20, 999: 30, 142: 40, 6582: 60}
    stoich_colous = {
        "9-9-9": "tab:green",
        "6-12-9": "tab:blue",
        "12-6-9": "tab:orange",
        "2-2-2": "gray",
        "1-3-2": "gray",
        "3-1-2": "gray",
        "3-3-3": "gray",
        "2-4-3": "gray",
        "4-2-3": "gray",
        "4-4-4": "gray",
        "5-5-5": "gray",
        "6-6-6": "gray",
        "4-8-6": "gray",
        "8-4-6": "gray",
        "7-7-7": "gray",
        "8-8-8": "tab:pink",
    }
    pair_markers = {
        "cs490_cs41c": "o",
        "cs490_cs41d": "X",
        "cs490_cs41a": "D",
        "cs490_cs41b": "P",
    }
    gen_entries = defaultdict(dict)
    total_min_energy = float("inf")

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        gid = entry.properties.get("generation_id", 0)
        gseed = entry.properties.get("generation_seed", 0)
        energy = entry.properties["energy_per_bb"]
        stoichstring = entry.properties["stoichstring"]
        total_min_energy = min((total_min_energy, energy))

        if gseed not in gen_entries[(pair, stoichstring)]:
            gen_entries[(pair, stoichstring)][gseed] = defaultdict(dict)
        if gid not in gen_entries[(pair, stoichstring)][gseed]:
            gen_entries[(pair, stoichstring)][gseed][gid] = []

        gen_entries[(pair, stoichstring)][gseed][gid].append(
            (entry.key, energy)
        )

    for (pair, stoichstring), byseed in gen_entries.items():
        xys = []
        min_energy = float("inf")
        min_energy_key = None
        for seed, offset in seeds.items():
            if seed not in byseed:
                continue
            bygen = byseed[seed]

            for gid in sorted(bygen.keys()):
                key, ey = min(bygen[gid], key=lambda i: i[1])

                if ey < min_energy:
                    min_energy = ey
                    min_energy_key = key
                xys.append((gid + offset, min_energy))
        logging.info("for %s, %s is min energy", stoichstring, min_energy_key)

        ax.plot(
            [i[0] for i in xys],
            [i[1] for i in xys],
            lw=2,
            c=stoich_colous[stoichstring],
            marker=pair_markers[pair],
            markersize=7,
            markeredgecolor="w",
            label=f"{pair}: {stoichstring}"
            if stoich_colous[stoichstring] != "gray"
            else None,
            zorder=2 if stoich_colous[stoichstring] != "gray" else -1,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.legend(fontsize=16)
    ax.axhline(y=total_min_energy, c="k", ls="--", zorder=-2)
    ax.axhspan(0, isomer_energy(), color="tab:grey", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlim(-1, 20 + 20 + 10 * 4)

    ax.legend(ncols=3, fontsize=10)

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


def plot_energies_main(  # noqa: C901, PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    chosen_pair: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 4))

    seeds = {4: 0, 12689: 10, 18: 20, 999: 30, 142: 40, 6582: 60}
    stoich_colous = {
        "9-9-9": "tab:green",
        "6-12-9": "tab:blue",
        "12-6-9": "tab:orange",
        "8-8-8": "tab:pink",
    }

    gen_entries = defaultdict(dict)
    total_min_energy = float("inf")

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair != chosen_pair:
            continue
        gid = entry.properties.get("generation_id", 0)
        gseed = entry.properties.get("generation_seed", 0)
        energy = entry.properties["energy_per_bb"]
        stoichstring = entry.properties["stoichstring"]
        if stoichstring not in stoich_colous:
            continue
        total_min_energy = min((total_min_energy, energy))

        if gseed not in gen_entries[(pair, stoichstring)]:
            gen_entries[(pair, stoichstring)][gseed] = defaultdict(dict)
        if gid not in gen_entries[(pair, stoichstring)][gseed]:
            gen_entries[(pair, stoichstring)][gseed][gid] = []

        gen_entries[(pair, stoichstring)][gseed][gid].append(
            (entry.key, energy)
        )

    for (_, stoichstring), byseed in gen_entries.items():
        xys = []
        min_energy = float("inf")
        min_key = None

        for seed, offset in seeds.items():
            if seed not in byseed:
                continue
            bygen = byseed[seed]

            for gid in sorted(bygen.keys()):
                key, ey = min(bygen[gid], key=lambda i: i[1])

                min_energy = min(ey, min_energy)
                min_key = key if ey == min_energy else min_key
                xys.append((gid + offset, min_energy, key))

        logging.info(
            "min of all generations is %s with Eb=%s",
            min_key,
            round(min_energy, 2),
        )

        ax.plot(
            [i[0] for i in xys],
            [i[1] for i in xys],
            lw=2,
            c=stoich_colous[stoichstring],
            marker="o",
            markersize=6,
            markeredgecolor="w",
            label=f"{stoichstring}"
            if stoich_colous[stoichstring] != "gray"
            else None,
            zorder=2 if stoich_colous[stoichstring] != "gray" else -1,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.legend(fontsize=16)
    ax.axhline(y=total_min_energy, c="k", ls="--", zorder=-2)
    ax.set_yscale("log")
    ax.set_xlim(-1, 20 + 20 + 10 * 4)
    ax.axvline(x=10, c="k", lw=1, alpha=0.2, zorder=-1)
    ax.axvline(x=20, c="k", lw=1, alpha=0.2, zorder=-1)
    ax.axvline(x=30, c="k", lw=1, alpha=0.2, zorder=-1)
    ax.axvline(x=40, c="k", lw=1, alpha=0.2, zorder=-1)
    ax.axvline(x=60, c="k", lw=1, alpha=0.2, zorder=-1)

    ax.legend(ncols=1, fontsize=16)

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


def plot_energy_by_seed(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    stoichstring: str,
    chosen_pair: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    seed_data = {u: [] for u in seed_cs}

    max_energy = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair != chosen_pair:
            continue
        if entry.properties["stoichstring"] != stoichstring:
            continue
        if "generation_seed" not in entry.properties:
            continue
        gseed = entry.properties["generation_seed"]
        energy = entry.properties["energy_per_bb"]
        seed_data[gseed].append(energy)
        max_energy = max((max_energy, energy))

    steps = range(len(seed_data) - 1, -1, -1)
    xmin = 0
    xmax = max_energy
    xwidth = 0.2
    ystep = 1
    xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)

    for i, value in enumerate(seed_cs):
        ax.hist(
            x=seed_data[value],
            bins=xbins,
            density=True,
            bottom=steps[i] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            facecolor=seed_cs[value],
            label=f"seed: {value}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(eb_str(), fontsize=16)

    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.set_ylim(0, (steps[0] + 1.5) * ystep)
    ax.legend(fontsize=16)
    ax.legend(ncols=3, fontsize=16)

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


def progress_plot(
    seeded_generations: dict[int, list],
    output: pathlib.Path,
) -> None:
    """Draw optimisation progress."""
    fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(8, 5))

    max_fitness = 0
    for i, (seed, generations) in enumerate(seeded_generations.items()):
        fitnesses = [
            generation.calculate_fitness_values() for generation in generations
        ]
        max_fitness = max((max_fitness, max([max(i) for i in fitnesses])))

        ax.plot(
            [max(i) for i in fitnesses],
            label=f"{seed}",
            lw=2,
            c=seed_cs[seed],
            marker="o",
            markersize=8,
            markeredgecolor="w",
        )
        ax.plot(
            [np.mean(i) for i in fitnesses], lw=2, ls="--", c=seed_cs[seed]
        )

        ax1.plot(
            [max(i) for i in fitnesses],
            label=f"{seed}",
            lw=2,
            c=seed_cs[seed],
            marker="o",
            markersize=8,
            markeredgecolor="w",
        )
        ax1.plot(
            [np.mean(i) for i in fitnesses],
            lw=2,
            ls="--",
            c=seed_cs[seed],
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_ylim(0, 1)
    ax.legend(ncols=2, fontsize=16)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("generation", fontsize=16)
    ax1.set_ylabel("fitness", fontsize=16)
    if max_fitness > 1e-10:  # noqa: PLR2004
        ax1.set_ylim(1e-10, 1)
    ax1.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output, dpi=360, bbox_inches="tight")
    fig.savefig(
        str(output).replace(".png", ".pdf"), dpi=360, bbox_inches="tight"
    )
    plt.close("all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def plot_fitness_curve(
    figure_dir: pathlib.Path,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    x = np.linspace(0, 10, 100)
    for beta in (0.1, 1, 2, 5, 10, 100):
        y = np.exp(-x * beta)
        ax.plot(x, y, label=beta)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("energy", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / "fitness.png", dpi=360, bbox_inches="tight")
    fig.savefig(figure_dir / "fitness.pdf", dpi=360, bbox_inches="tight")
    plt.close()


def define_pairs(
    pairs_to_predict: abc.Sequence[tuple[abc.Sequence[str]]],
    ligand_types: dict[str, str],
) -> dict[str, dict[str, cgx.molecular.Precursor | tuple | int]]:
    """Define pairs from provided information."""
    pairs = {}
    for large, small in pairs_to_predict:
        name = f"{large}_{small}"

        if ligand_types[large] == "sixbead":
            large_prec = cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            )
        elif ligand_types[large] == "twoarm":
            large_prec = cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c)

        else:
            msg = large
            raise NotImplementedError(msg)

        if ligand_types[small] == "twoarm":
            small_prec = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)

        elif ligand_types[small] == "sixbead":
            small_prec = cgx.molecular.SixBead(
                bead=c2bead_d,
                abead1=a2bead_d,
                abead2=e2bead_d,
            )
        else:
            msg = small
            raise NotImplementedError(msg)

        pairs[name] = {
            "large_name": large,
            "small_name": small,
            "large": large_prec,
            "small": small_prec,
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "vdw_cutoff": 2,
        }

    return pairs


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run case study studying PW heteroleptic systems."""
    run = _parse_args().run

    wd = pathlib.Path("/home/tarziaa/workingspace/skewed_production/")
    run_prefix = "cu18"
    calculation_dir = wd / f"{run_prefix}_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / f"{run_prefix}_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / f"{run_prefix}_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / f"{run_prefix}_data"
    data_dir.mkdir(exist_ok=True)
    (wd / "figures").mkdir(exist_ok=True)
    figure_dir = wd / "figures" / f"{run_prefix}"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / f"{run_prefix}.db"
    plot_fitness_curve(figure_dir)

    stoichiometries_l_s_m = set()
    for n_tetra in range(2, 10):
        n_ditopic = 2 * n_tetra

        for m in range(n_ditopic):
            stoichiometries_l_s_m.add((m, n_ditopic - m, n_tetra))
            stoichiometries_l_s_m.add((n_ditopic - m, m, n_tetra))

    stoichiometries_l_s_m = sorted(
        stoichiometries_l_s_m,
        key=lambda stoichiometries_l_s_m: stoichiometries_l_s_m[2],
    )
    logging.info("there are %s stoichiometries", len(stoichiometries_l_s_m))

    ligand_measures = {
        "cs41a": {"ba": 1.5, "aa": 9.5, "bac": 145},
        "cs490": {"ba": 1.5, "aa": 5.9, "bac": 135},
    }
    ligand_types = {
        "cs41a": "twoarm",
        "cs41b": "twoarm",
        "cs41c": "twoarm",
        "cs41d": "twoarm",
        "cs490": "twoarm",
    }
    pairs_to_predict = [
        # large, small.
        ("cs490", "cs41a"),
    ]

    pairs = define_pairs(pairs_to_predict, ligand_types)

    cs4_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbe": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "ede": ("angle", 180, 1e2),
        # Torsions.
        "bacab": ("tors", "0134", 180, 50, 1),
        "bedeb": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
    }

    short_scan_config = {
        "seeds": [4, 12689, 18, 999],
        "count_to_spam": 2,
        "mutations": 2,
        "num_generations": 10,
        "selection_size": 10,
        "num_processes": 1,
    }

    long_scan_config = {
        "seeds": [142],
        "count_to_spam": 5,
        "mutations": 10,
        "num_generations": 20,
        "selection_size": 20,
        "num_processes": 1,
    }

    if run:
        for pair in pairs:
            forcefield = precursors_to_forcefield(
                pair=pair,
                large=pairs[pair]["large"],
                small=pairs[pair]["small"],
                large_meas=ligand_measures[pairs[pair]["large_name"]],
                small_meas=ligand_measures[pairs[pair]["small_name"]],
                vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
                constant_definer_dict=cs4_definer_dict,
            )

            small_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["small"].get_building_block(),
                name=f"{pair}_{pairs[pair]['small'].get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            small_bb.write(
                str(
                    ligand_dir
                    / f"{pair}_{pairs[pair]['small'].get_name()}_optl.mol"
                )
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["tetra"].get_building_block(),
                name=pairs[pair]["tetra"].get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(
                str(ligand_dir / f"{pairs[pair]['tetra'].get_name()}_optl.mol")
            )

            large_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["large"].get_building_block(),
                name=f"{pair}_{pairs[pair]['large'].get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            large_bb.write(
                str(
                    ligand_dir
                    / f"{pair}_{pairs[pair]['large'].get_name()}_optl.mol"
                )
            )

            for stoichiometry_l_s_m in stoichiometries_l_s_m:
                stoichstring = "-".join([str(i) for i in stoichiometry_l_s_m])

                logging.info(
                    "doing: pair %s, stoichstring %s", pair, stoichstring
                )

                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.SkewedTopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichiometry_l_s_m[2],
                        large_bb: stoichiometry_l_s_m[0],
                        small_bb: stoichiometry_l_s_m[1],
                    },
                )
                all_topology_codes = tuple(enumerate(iterator.yield_graphs()))
                topology_codes = []
                for tidx, tc in all_topology_codes:
                    if tc.contains_parallels():
                        continue
                    topology_codes.append((tidx, tc))

                logging.info(
                    "graph iteration has %s graphs (from %s)",
                    len(topology_codes),
                    len(all_topology_codes),
                )
                if len(topology_codes) != len(all_topology_codes):
                    raise RuntimeError

                possible_bbdicts = iterator.get_configurations()
                logging.info(
                    "building block iteration has %s options",
                    len(possible_bbdicts),
                )

                chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
                    prefix=f"{pair}_{stoichstring}",
                    present_beads=forcefield.get_present_beads(),
                    vdw_bond_cutoff=forcefield.get_vdw_bond_cutoff(),
                )
                chromo_it.add_gene(
                    iteration=topology_codes,
                    gene_type="topology",
                )
                chromo_it.add_gene(
                    iteration=possible_bbdicts,
                    # This gene type is out of date now.
                    gene_type="building_block_configuration",
                )

                # Define fitness calculator.
                fitness_calculator = (
                    cgx.systems_optimisation.FitnessCalculator(
                        fitness_function=fitness_function,
                        chromosome_generator=chromo_it,
                        structure_output=structure_dir,
                        calculation_output=calculation_dir,
                        database_path=database_path,
                        options={"beta": 5},
                    )
                )

                # Define structure calculator.
                structure_calculator = (
                    cgx.systems_optimisation.StructureCalculator(
                        structure_function=structure_function,
                        structure_output=structure_dir,
                        calculation_output=calculation_dir,
                        database_path=database_path,
                        options={
                            "topology_codes": list(all_topology_codes),
                            "bb_configs": possible_bbdicts,
                            "iterator": iterator,
                            "forcefield": forcefield,
                            "attempts": attempts,
                        },
                    )
                )

                timing_file = (
                    data_dir / f"np_{short_scan_config['num_processes']}.txt"
                )
                seeded_generations = {}
                for seed in short_scan_config["seeds"]:
                    seeded_generations[seed] = []
                    generator = np.random.default_rng(seed)

                    initial_population = chromo_it.select_random_population(
                        generator,
                        size=short_scan_config["selection_size"],
                    )

                    # Yield this.
                    generations = []
                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=initial_population,
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=short_scan_config["num_processes"],
                    )

                    generation.run_structures()
                    _ = generation.calculate_fitness_values()
                    generations.append(generation)
                    seeded_generations[seed].append(generation)

                    # Add generational information.
                    add_generation_information(
                        database_path=database_path,
                        generation=generation,
                        seed=seed,
                        generation_id=0,
                    )
                    best_chromosome = generation.select_best(selection_size=1)[
                        0
                    ]

                    count_unchanged = 0
                    previous_best = (
                        f"{best_chromosome.prefix}"
                        f"_{best_chromosome.get_topology_information()[0]}_"
                        f"b{best_chromosome.get_building_block_configurations()[0].idx}"
                    )
                    for generation_id in range(
                        1, short_scan_config["num_generations"] + 1
                    ):
                        logging.info(
                            "doing generation %s of seed %s",
                            generation_id,
                            seed,
                        )

                        logging.info(
                            "initial size is %s.",
                            generation.get_generation_size(),
                        )
                        if (
                            count_unchanged
                            == short_scan_config["count_to_spam"]
                        ):
                            count_unchanged = 0
                            logging.info("doing 2x mutations.")
                            num_mutate = short_scan_config["mutations"] * 2
                        else:
                            logging.info("doing mutations.")
                            num_mutate = short_scan_config["mutations"] * 1

                        merged_chromosomes = []
                        chromosome_list = {
                            (
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_topology_information()[0]}"
                                f"_b{chromosome.get_building_block_configurations()[0].idx}"
                            ): chromosome
                            for chromosome in generation.chromosomes
                        }
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="random",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                gene_range=chromo_it.get_bc_ids(),
                                selection="random",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            chromo_it.crossover_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                selection="random",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            roulette_mutate_population(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_building_block_configurations()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            roulette_mutate_population(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_building_block_configurations()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                generator=generator,
                                gene_range=chromo_it.get_bc_ids(),
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            roulette_crossover_population(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_building_block_configurations()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                generator=generator,
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        # Add the best 5 to the new generation.
                        merged_chromosomes.extend(
                            generation.select_best(selection_size=num_mutate)
                        )

                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(
                                merged_chromosomes
                            ),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=short_scan_config["num_processes"],
                        )
                        logging.info(
                            "new size is %s.", generation.get_generation_size()
                        )

                        # Build, optimise and analyse each structure.
                        st = time.time()
                        generation.run_structures()
                        str_time = time.time() - st
                        st = time.time()
                        _ = generation.calculate_fitness_values()
                        fit_time = time.time() - st
                        with timing_file.open("a") as f:
                            f.write(f"{str_time},{fit_time}\n")

                        # Add generational information.
                        add_generation_information(
                            database_path=database_path,
                            generation=generation,
                            seed=seed,
                            generation_id=generation_id,
                        )

                        # Add final state to generations.
                        generations.append(generation)
                        seeded_generations[seed].append(generation)

                        # Select the best of the generation for the next
                        # generation.
                        best = generation.select_best(
                            selection_size=short_scan_config["selection_size"]
                        )
                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(best),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=short_scan_config["num_processes"],
                        )
                        logging.info(
                            "final size is %s.",
                            generation.get_generation_size(),
                        )

                        progress_plot(
                            seeded_generations=seeded_generations,
                            output=figure_dir
                            / f"fp_{pair}_{stoichstring}.png",
                        )

                        # Output best structures as images.
                        best_chromosome = generation.select_best(
                            selection_size=1
                        )[0]

                        best_name = (
                            f"{best_chromosome.prefix}"
                            f"_{best_chromosome.get_topology_information()[0]}_"
                            f"b{best_chromosome.get_building_block_configurations()[0].idx}"
                        )
                        if best_name != previous_best:
                            count_unchanged = 0
                            previous_best = best_name
                        else:
                            count_unchanged += 1

                        logging.info(
                            "top scorer is %s (seed: %s)", best_name, seed
                        )

                # Report.
                found = set()
                for generation in generations:
                    for chromo in generation.chromosomes:
                        found.add(chromo.name)

                logging.info(
                    "%s chromosomes found in EA (of %s)",
                    len(found),
                    chromo_it.get_num_chromosomes(),
                )

                # Run longer GA from elites.
                chromosomes = []
                for generations in seeded_generations.values():
                    for generation in generations:
                        chromosomes.extend(generation.chromosomes)

                elite_population = cgx.systems_optimisation.Generation(
                    chromosomes=chromo_it.dedupe_population(chromosomes),
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    num_processes=long_scan_config["num_processes"],
                )

                timing_file = (
                    data_dir / f"np_{long_scan_config['num_processes']}.txt"
                )
                for seed in long_scan_config["seeds"]:
                    seeded_generations[seed] = []
                    generator = np.random.default_rng(seed)

                    initial_population = elite_population.select_elite(
                        proportion_threshold=0.25
                    )

                    logging.info(
                        "selected elite with f>%s",
                        round(
                            elite_population.calculate_elite_fitness(
                                proportion_threshold=0.25
                            ),
                            5,
                        ),
                    )

                    # Yield this.
                    generations = []
                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=initial_population,
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=long_scan_config["num_processes"],
                    )

                    generation.run_structures()
                    _ = generation.calculate_fitness_values()
                    generations.append(generation)
                    seeded_generations[seed].append(generation)

                    # Add generational information.
                    add_generation_information(
                        database_path=database_path,
                        generation=generation,
                        seed=seed,
                        generation_id=0,
                    )
                    best_chromosome = generation.select_best(selection_size=1)[
                        0
                    ]

                    count_unchanged = 0
                    previous_best = (
                        f"{best_chromosome.prefix}"
                        f"_{best_chromosome.get_topology_information()[0]}_"
                        f"b{best_chromosome.get_building_block_configurations()[0].idx}"
                    )
                    for generation_id in range(
                        1, long_scan_config["num_generations"] + 1
                    ):
                        logging.info(
                            "doing generation %s of seed %s",
                            generation_id,
                            seed,
                        )

                        logging.info(
                            "initial size is %s.",
                            generation.get_generation_size(),
                        )
                        if (
                            count_unchanged
                            == long_scan_config["count_to_spam"]
                        ):
                            count_unchanged = 0
                            logging.info("doing 2x mutations.")
                            num_mutate = long_scan_config["mutations"] * 2
                        else:
                            logging.info("doing mutations.")
                            num_mutate = long_scan_config["mutations"] * 1

                        merged_chromosomes = []
                        chromosome_list = {
                            (
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_topology_information()[0]}"
                                f"_b{chromosome.get_building_block_configurations()[0].idx}"
                            ): chromosome
                            for chromosome in generation.chromosomes
                        }
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="random",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                gene_range=chromo_it.get_bc_ids(),
                                selection="random",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            chromo_it.crossover_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                selection="random",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                gene_range=chromo_it.get_bc_ids(),
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            chromo_it.crossover_population(
                                chromosomes=chromosome_list,
                                generator=generator,
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        # Add the best 5 to the new generation.
                        merged_chromosomes.extend(generation.select_all())

                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(
                                merged_chromosomes
                            ),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=long_scan_config["num_processes"],
                        )
                        logging.info(
                            "new size is %s.", generation.get_generation_size()
                        )

                        # Build, optimise and analyse each structure.
                        st = time.time()
                        generation.run_structures()
                        str_time = time.time() - st
                        st = time.time()
                        _ = generation.calculate_fitness_values()
                        fit_time = time.time() - st
                        with timing_file.open("a") as f:
                            f.write(f"{str_time},{fit_time}\n")

                        # Add generational information.
                        add_generation_information(
                            database_path=database_path,
                            generation=generation,
                            seed=seed,
                            generation_id=generation_id,
                        )

                        # Add final state to generations.
                        generations.append(generation)
                        seeded_generations[seed].append(generation)

                        # Select the best of the generation for the next
                        # generation.
                        best = generation.select_best(
                            selection_size=long_scan_config["selection_size"]
                        )
                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(best),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=long_scan_config["num_processes"],
                        )
                        logging.info(
                            "final size is %s.",
                            generation.get_generation_size(),
                        )

                        progress_plot(
                            seeded_generations=seeded_generations,
                            output=figure_dir
                            / f"fp_{pair}_{stoichstring}.png",
                        )

                        # Output best structures as images.
                        best_chromosome = generation.select_best(
                            selection_size=1
                        )[0]

                        best_name = (
                            f"{best_chromosome.prefix}"
                            f"_{best_chromosome.get_topology_information()[0]}_"
                            f"b{best_chromosome.get_building_block_configurations()[0].idx}"
                        )
                        if best_name != previous_best:
                            count_unchanged = 0
                            previous_best = best_name
                        else:
                            count_unchanged += 1

                        logging.info(
                            "top scorer is %s (seed: %s)", best_name, seed
                        )

                # Run neighbour scan over top candidates.
                chromosomes = []
                for generations in seeded_generations.values():
                    for generation in generations:
                        chromosomes.extend(generation.chromosomes)

                elite_population = cgx.systems_optimisation.Generation(
                    chromosomes=chromo_it.dedupe_population(chromosomes),
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    num_processes=long_scan_config["num_processes"],
                )

                timing_file = (
                    data_dir / f"np_{long_scan_config['num_processes']}.txt"
                )
                n_scan_config = {
                    "seeds": [6582],
                    "num_processes": 1,
                    "num_generations": 20,
                    "selection_size": 200,
                }
                for seed in n_scan_config["seeds"]:
                    seeded_generations[seed] = []

                    initial_population = elite_population.select_elite(
                        proportion_threshold=0.10
                    )
                    if len(initial_population) == 0:
                        initial_population = elite_population.select_elite(
                            proportion_threshold=0.50
                        )

                    # Yield this.
                    generations = []
                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=initial_population,
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=n_scan_config["num_processes"],
                    )

                    generation.run_structures()
                    _ = generation.calculate_fitness_values()
                    generations.append(generation)
                    seeded_generations[seed].append(generation)

                    # Add generational information.
                    add_generation_information(
                        database_path=database_path,
                        generation=generation,
                        seed=seed,
                        generation_id=0,
                    )

                    for generation_id in range(
                        1, n_scan_config["num_generations"] + 1
                    ):
                        logging.info(
                            "doing generation %s of seed %s",
                            generation_id,
                            seed,
                        )
                        logging.info(
                            "initial size is %s.",
                            generation.get_generation_size(),
                        )

                        merged_chromosomes = []
                        merged_chromosomes.extend(
                            get_population_neighbours(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_building_block_configurations()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                selection="all",
                                gene_range=chromo_it.get_bc_ids(),
                            )
                        )

                        # Add the best 5 to the new generation.
                        merged_chromosomes.extend(generation.select_all())

                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(
                                merged_chromosomes
                            ),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=n_scan_config["num_processes"],
                        )
                        logging.info(
                            "new size is %s.", generation.get_generation_size()
                        )

                        # Build, optimise and analyse each structure.
                        st = time.time()
                        generation.run_structures()
                        str_time = time.time() - st
                        st = time.time()
                        _ = generation.calculate_fitness_values()
                        fit_time = time.time() - st
                        with timing_file.open("a") as f:
                            f.write(f"{str_time},{fit_time}\n")

                        # Add generational information.
                        add_generation_information(
                            database_path=database_path,
                            generation=generation,
                            seed=seed,
                            generation_id=generation_id,
                        )

                        # Add final state to generations.
                        generations.append(generation)
                        seeded_generations[seed].append(generation)

                        # Select the best of the generation for the next
                        # generation.
                        best = generation.select_best(
                            selection_size=n_scan_config["selection_size"]
                        )
                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(best),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=n_scan_config["num_processes"],
                        )
                        logging.info(
                            "final size is %s.",
                            generation.get_generation_size(),
                        )

                        progress_plot(
                            seeded_generations=seeded_generations,
                            output=figure_dir
                            / f"fp_{pair}_{stoichstring}.png",
                        )

                        # Output best structures as images.
                        best_chromosome = generation.select_best(
                            selection_size=1
                        )[0]

                        best_name = (
                            f"{best_chromosome.prefix}"
                            f"_{best_chromosome.get_topology_information()[0]}_"
                            f"b{best_chromosome.get_building_block_configurations()[0].idx}"
                        )

                        logging.info(
                            "top scorer is %s (seed: %s)", best_name, seed
                        )

    for pair in pairs:
        plot_energies_main(
            database_path=database_path,
            figure_dir=figure_dir,
            filename=f"cu18_2m_{pair}.png",
            chosen_pair=pair,
        )

        for stoichiometry_l_s_m in stoichiometries_l_s_m:
            stoichstring = "-".join([str(i) for i in stoichiometry_l_s_m])
            logging.info("plotting %s", stoichstring)
            plot_elite_graphs(
                database_path=database_path,
                stoichstring=stoichstring,
                figure_dir=figure_dir,
                filename=f"cu18_12_{stoichstring}_{pair}.png",
                chosen_pair=pair,
            )

            plot_counters(
                database_path=database_path,
                stoichstring=stoichstring,
                figure_dir=figure_dir,
                filename=f"cu18_1_{stoichstring}_{pair}.png",
                chosen_pair=pair,
            )
            plot_vs_parallels(
                database_path=database_path,
                stoichstring=stoichstring,
                figure_dir=figure_dir,
                filename=f"cu18_9_{stoichstring}_{pair}.png",
                chosen_pair=pair,
            )

            plot_idx_completion(
                database_path=database_path,
                stoichstring=stoichstring,
                figure_dir=figure_dir,
                filename=f"cu18_10_{stoichstring}_{pair}.png",
                chosen_pair=pair,
            )
            plot_energy_by_seed(
                database_path=database_path,
                stoichstring=stoichstring,
                figure_dir=figure_dir,
                filename=f"cu18_6_{stoichstring}_{pair}.png",
                chosen_pair=pair,
            )

    plot_energies(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="cu18_2.png",
    )

    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="cu18_5.png",
    )

    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="cu18_4.png",
        pairs=tuple((i, None) for i in pairs_to_predict),
        structure_dir=structure_dir,
    )
    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="cu18_3.png",
        pairs=tuple((i, None) for i in pairs_to_predict),
        width_height=(16, 5),
    )
    plot_timings(figure_dir, data_dir)


if __name__ == "__main__":
    main()
