#!/usr/bin/env python

import shutil
import click
import os
import python.helpers as helpers
from python.helpers import print_v
from python.propagation_data import PropagationData
from python.run import sample_inequalities, run_with_inequalities


@click.group()
@click.option("--steps", default=100)
@click.option("--step-size", default=1)
@click.option("--block-size", default=None)
@click.option("--plot/--no-plot", default=False)
@click.option("--save-path", default="data")
@click.option("--save-dir", default=None)
@click.option("--save-data/--no-save-data", default=True)
@click.option("--pack-data/--no-pack-data", default=False)
@click.option("--delete-uncompressed/--no-delete-uncompressed", default=False)
@click.option("--archive-type", default="gztar")
@click.option("--plot-type", default="pdf")
@click.option("--plot-max", default=None, type=int)
@click.option("--histogram-bins", default=50, type=int)
@click.option("--run-reduction/--estimate-only", default=False)
@click.option("--max-beta", default=30, type=int)
@click.option("--additional-fplll-params", default="", type=str)
@click.option("--use-best-step/--use-last-step", default=True)
@click.option("--max-enumerate", default=1, type=int)
@click.option("--verbose/--silent", default=True)
@click.pass_context
def main(
    ctx,
    steps,
    step_size,
    block_size,
    plot,
    save_path,
    save_dir,
    save_data,
    pack_data,
    delete_uncompressed,
    archive_type,
    plot_type,
    plot_max,
    histogram_bins,
    run_reduction,
    max_beta,
    additional_fplll_params,
    use_best_step,
    max_enumerate,
    verbose,
):
    ctx.ensure_object(dict)
    ctx.obj["steps"] = steps
    ctx.obj["step_size"] = step_size
    ctx.obj["block_size"] = block_size
    ctx.obj["plot"] = plot
    ctx.obj["save_data"] = save_data
    ctx.obj["pack_data"] = pack_data
    ctx.obj["delete_uncompressed"] = delete_uncompressed
    ctx.obj["archive_type"] = archive_type
    ctx.obj["plot_type"] = plot_type
    ctx.obj["histogram_bins"] = histogram_bins
    ctx.obj["plot_max"] = plot_max
    ctx.obj["run_reduction"] = run_reduction
    ctx.obj["max_beta"] = max_beta
    ctx.obj["additional_fplll_params"] = additional_fplll_params
    ctx.obj["use_best_step"] = use_best_step
    ctx.obj["max_enumerate"] = max_enumerate
    ctx.obj["save_path"] = save_path
    ctx.obj["save_dir"] = save_dir
    helpers.set_verbose(verbose)


@main.command()
@click.pass_context
@click.argument("path")
@click.option("--load-ineqs/--no-load-ineqs", default=False)
@click.option("--load-steps/--no-load-steps", default=False)
@click.option("--load-last/--no-load-last", default=True)
def load(ctx, path, load_ineqs, load_steps, load_last):
    steps = ctx.obj["steps"]
    step_size = ctx.obj["step_size"]
    plot = ctx.obj["plot"]
    save_data = ctx.obj["save_data"]
    pack_data = ctx.obj["pack_data"]
    delete_uncompressed = ctx.obj["delete_uncompressed"]
    archive_type = ctx.obj["archive_type"]
    plot_type = ctx.obj["plot_type"]
    histogram_bins = ctx.obj["histogram_bins"]
    plot_max = ctx.obj["plot_max"]
    propagation_data = PropagationData.load_data(
        path, load_ineqs, load_steps, load_last
    )
    success, propagation_data = run_with_inequalities(
        propagation_data,
        steps,
        step_size,
        ctx.obj["block_size"],
        run_reduction=ctx.obj["run_reduction"],
        max_beta=ctx.obj["max_beta"],
        add_fplll=ctx.obj["additional_fplll_params"],
        use_best_step=ctx.obj["use_best_step"],
        max_enum=ctx.obj["max_enumerate"],
    )
    propagation_data.set_settings(ctx.obj)
    if propagation_data.inequalities:
        filtered_ratio_run = len(propagation_data.inequalities) / (
            len(propagation_data.inequalities) + propagation_data.filtered_cts
        )
        print_v(
            f"Success: {success}; filtered_cts: {propagation_data.filtered_cts}; used cts ratio: {filtered_ratio_run:.2f}"
        )
    else:
        print_v(f"Success: {success}")
    process_propagation_data(
        propagation_data,
        plot,
        save_data,
        pack_data,
        delete_uncompressed,
        plot_type,
        plot_max,
        histogram_bins,
        archive_type,
        ctx.obj["save_path"],
        ctx.obj["save_dir"],
    )


@main.command()
@click.pass_context
@click.argument("number_inequalities", type=int)
@click.option("--max-delta-v", default=None, type=int)
@click.option("--p-correct", default=1.0, type=float)
@click.option("--multiple-runs", default=1, type=int)
@click.option("--certain-correct", default=None, type=int)
def new(
    ctx,
    number_inequalities,
    max_delta_v,
    p_correct,
    multiple_runs,
    certain_correct,
):
    steps = ctx.obj["steps"]
    step_size = ctx.obj["step_size"]
    plot = ctx.obj["plot"]
    save_data = ctx.obj["save_data"]
    pack_data = ctx.obj["pack_data"]
    delete_uncompressed = ctx.obj["delete_uncompressed"]
    archive_type = ctx.obj["archive_type"]
    plot_type = ctx.obj["plot_type"]
    histogram_bins = ctx.obj["histogram_bins"]
    plot_max = ctx.obj["plot_max"]
    n = multiple_runs
    number_faults = number_inequalities
    assert not certain_correct or certain_correct == number_faults or p_correct < 1.0
    if p_correct >= 1.0:
        certain_correct = number_faults
    assert certain_correct <= number_faults
    sr = 0
    filtered_cts_avg = 0
    filtered_ratio_avg = 0
    for i in range(n):
        if i > 0:
            print_v("")
        print_v(f"Run {i}:")
        propagation_data = sample_inequalities(
            number_faults,
            p_correct,
            max_delta_v=max_delta_v,
            num_certain_correct=certain_correct,
        )
        success, propagation_data = run_with_inequalities(
            propagation_data,
            steps,
            step_size,
            ctx.obj["block_size"],
            ctx.obj["run_reduction"],
            ctx.obj["max_beta"],
            add_fplll=ctx.obj["additional_fplll_params"],
            use_best_step=ctx.obj["use_best_step"],
            max_enum=ctx.obj["max_enumerate"],
        )
        propagation_data.set_settings(ctx.obj)
        filtered_ratio_run = number_faults / (
            number_faults + propagation_data.filtered_cts
        )
        print_v(
            f"Success: {success}; filtered_cts: {propagation_data.filtered_cts}; used cts ratio: {filtered_ratio_run:.2f}"
        )
        process_propagation_data(
            propagation_data,
            plot,
            save_data,
            pack_data,
            delete_uncompressed,
            plot_type,
            plot_max,
            histogram_bins,
            archive_type,
            ctx.obj["save_path"],
            ctx.obj["save_dir"],
        )
        filtered_cts_avg += propagation_data.filtered_cts
        filtered_ratio_avg += filtered_ratio_run
        if success:
            sr += 1
        print_v("-" * 5)
    if n > 1:
        sr /= n
        print_v("")
        print_v(f"Success rate: {sr}")
        filtered_cts_avg /= n
        filtered_ratio_avg /= n
        print_v(f"Filtered cts average: {filtered_cts_avg}")
        print_v(f"Filtered cts ratio: {filtered_ratio_avg:.2f}")


def process_propagation_data(
    data,
    plot,
    save_data,
    pack_data,
    delete_uncompressed,
    plot_type,
    plot_max,
    histogram_bins,
    archive_type,
    save_path,
    save_dir,
):
    if save_data:
        if save_path:
            data.set_dir_prefix(save_path)
        if save_dir:
            data.set_dir(save_dir)
        data.save_data()
    if plot:
        data.plot(plot_type=plot_type, bins=histogram_bins, plot_max=plot_max)
    if (plot or save_data) and pack_data:
        print_v("Packing archive..")
        os.makedirs("archives", exist_ok=True)
        shutil.make_archive(
            f"archives/{data.dir.replace('/', '_')}", archive_type, data.dir
        )
        if delete_uncompressed:
            print_v("Removing uncompressed data..")
            shutil.rmtree(data.dir)


if __name__ == "__main__":
    main(obj={})
