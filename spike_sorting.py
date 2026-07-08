import os
import sys
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import networkx as nx
import torch
import argparse
import quantities as pq
torch.cuda.empty_cache()

os.environ["LD_LIBRARY_PATH"] = "/home/mxwbio/MaxLab/so"
os.environ["HDF5_PLUGIN_PATH"] = "/home/mxwbio/MaxLab/so"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import spikeinterface.full as si
import spikeinterface.sorters as ss
from scipy.spatial import ConvexHull

from utils.spike_sorting_utils import (
    # Settings / output
    load_analysis_settings,

    # Cleanup / conversion helpers
    remove_folder_if_requested,
    cast_to_float32,
    unsigned_to_signed_safe,

    # Diagnostics / preprocessing
    diagnose_recording_scaling,
    remove_zero_noise_channels,
    detect_mad_bad_channels,

    # Kilosort
    run_kilosort4_fresh,

    # Feature extraction
    compute_waveform_spatial_features,
    make_unit_locations_df,
    compute_quality_metrics_safe,
    compute_single_unit_firing_features,
    compute_burst_features,
    compute_max_interval_burst_features,
    compute_windowed_unit_features,
    make_windowed_unit_features_wide,
    detect_network_bursts,
    compute_surrogate_tested_sttc,
    compute_graph_features,

    # Final summaries
    make_well_summary,
)


print("Libraries imported and path verified! Ready to process wells.")

def main(project_name: str, date_label: str, plate_label: str) -> None:
    data_root = Path("../Data") / project_name / date_label / plate_label / "Network"
    raw_files = sorted(data_root.glob("*/data.raw.h5"))

    output_root = (
        Path("../Results")
        / f"{project_name}_{date_label}_{plate_label}"
    )

    kilosorter_output_dir = output_root / "kilosorter_outputs"
    diagnostics_dir = output_root / "diagnostics"
    phenotype_dir = output_root / "well_level_phenotype_matrices"

    phenotype_dir.mkdir(parents=True, exist_ok=True)
    kilosorter_output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)


    ##### LOAD SETTINGS
    settings = load_analysis_settings(args.settings_path)

    UNSIGNED_BIT_DEPTH = settings["unsigned_bit_depth"]
    FOOTPRINT_RELATIVE_THRESHOLD = settings["waveform_spatial_footprint"]["footprint_relative_threshold"]
    FOOTPRINT_MIN_P2P_UV = settings["waveform_spatial_footprint"]["footprint_min_p2p_uv"]
    MAX_INTERVAL_BURST_PARAMS = settings["max_interval_burst_detection"]
    NETWORK_BURST_BIN_S = settings["network_burst_detection"]["bin_s"]
    NETWORK_BURST_SMOOTH_BINS = settings["network_burst_detection"]["smooth_bins"]
    NETWORK_BURST_THRESHOLD_MAD = settings["network_burst_detection"]["threshold_mad"]
    NETWORK_BURST_MIN_UNITS_ABS = settings["network_burst_detection"]["min_units_abs"]
    NETWORK_BURST_MIN_UNITS_FRAC = settings["network_burst_detection"]["min_units_frac"]
    NETWORK_BURST_MIN_DURATION_S = settings["network_burst_detection"]["min_duration_s"]
    NETWORK_BURST_MERGE_GAP_S = settings["network_burst_detection"]["merge_gap_s"]
    STTC_DT_S = settings["sttc"]["dt_s"]
    STTC_N_SHUFFLES = settings["sttc"]["n_shuffles"]
    STTC_ALPHA = settings["sttc"]["alpha"]
    STTC_MIN_SPIKES_PER_UNIT = settings["sttc"]["min_spikes_per_unit"]
    CONNECTIVITY_MAX_UNITS = settings["sttc"]["connectivity_max_units"]
    SAVE_DENSE_STTC_MATRIX = settings["sttc"]["save_dense_matrix"]
    STTC_PROGRESS_EVERY_PAIRS = settings["sttc"]["progress_every_pairs"]
    DELETE_CACHE_AFTER_SUCCESS = settings["cleanup"]["delete_cache_after_success"]
    DELETE_ANALYZER_AFTER_SUCCESS = settings["cleanup"]["delete_analyzer_after_success"]
    DELETE_KILOSORT_OUTPUT_AFTER_SUCCESS = settings["cleanup"]["delete_kilosort_output_after_success"]
    DELETE_CACHE_WHEN_NO_UNITS = settings["cleanup"]["delete_cache_when_no_units"]
    DELETE_KILOSORT_WHEN_NO_UNITS = settings["cleanup"]["delete_kilosort_when_no_units"]
    PERTURBATION_WINDOWS = settings["perturbation_windows"]

    all_master_dfs = []
    use_assay_in_safe_id = len(raw_files) > 1 #to differentiate between two recordings on the same plate, same day

    for raw_file in raw_files:

        assay_number = raw_file.parent.name

        print(f"\n{'#' * 100}")
        print(f"PROCESSING RAW FILE: {raw_file}")
        print(f"OUTPUT_DIR: {output_root}")
        print(f"{'#' * 100}")

        with h5py.File(raw_file, "r") as f:
            recording_names = list(f["recordings"].keys())

        print(" Found recordings:", recording_names)

        # ==========================================
        ##PROCESS RECORDINGS / WELLS
        # ==========================================

        for TARGET_REC in recording_names:

            print(f"\n{'=' * 80}")
            print(f"PROCESSING RECORDING {TARGET_REC}")
            print(f"{'=' * 80}")

            stream_ids, stream_names = si.get_neo_streams(
                "maxwell",
                raw_file,
                rec_name=TARGET_REC
            )

            for stream_id, stream_name in zip(stream_ids, stream_names):

                well_name = stream_name.split(".")[0]
                if use_assay_in_safe_id:
                    safe_id = f"assay{assay_number}_{TARGET_REC}_{well_name}"
                else:
                    safe_id = f"{TARGET_REC}_{well_name}"

                print(f"\nProcessing {safe_id}")

                # Resume/skip logic: if this well's final phenotype file already exists,
                # skip the well instead of rerunning preprocessing, Kilosort, analyzer,
                # and STTC/network steps.
                final_well_csv = os.path.join(
                    phenotype_dir,
                    f"{safe_id}_phenotype_features.csv"
                )

                if os.path.exists(final_well_csv):
                    print(f"[{safe_id}] Final phenotype file already exists. Skipping.")
                    continue

                try:
                    # ==========================================
                    # Read recording
                    # ==========================================

                    recording = si.read_maxwell(
                        file_path=raw_file,
                        stream_id=stream_id,
                        rec_name=TARGET_REC,
                    )

                    diagnose_recording_scaling(
                        recording,
                        label=f"{safe_id}_raw_reader",
                        output_dir=diagnostics_dir,
                        seconds=2,
                    )

                    # ==========================================
                    # Correct unsigned conversion
                    # ==========================================

                    recording = unsigned_to_signed_safe(
                        recording,
                        safe_id=safe_id,
                        bit_depth=UNSIGNED_BIT_DEPTH,
                    )

                    diagnose_recording_scaling(
                        recording,
                        label=f"{safe_id}_after_unsigned_to_signed_bitdepth{UNSIGNED_BIT_DEPTH}",
                        output_dir=diagnostics_dir,
                        seconds=2,
                    )

                    recording = cast_to_float32(recording)

                    diagnose_recording_scaling(
                        recording,
                        label=f"{safe_id}_signed_float32",
                        output_dir=diagnostics_dir,
                        seconds=2,
                    )

                    t_stop = recording.get_total_duration() * pq.s
                    recording_duration_s = float(recording.get_total_duration())
                    sampling_rate = recording.get_sampling_frequency()

                    print(f"[{safe_id}] duration: {recording_duration_s:.2f} s")
                    print(f"[{safe_id}] sampling rate: {sampling_rate}")
                    print(f"[{safe_id}] channels initially: {len(recording.get_channel_ids())}")

                    # ==========================================
                    # Noise / bad-channel cleanup before sorting
                    # ==========================================

                    recording, raw_zero_noise_report = remove_zero_noise_channels(
                        recording=recording,
                        label=f"{safe_id}_raw",
                        output_dir=diagnostics_dir,
                        noise_floor_abs=1e-12,
                        noise_floor_rel=1e-6,
                        chunk_s=0.5,
                        num_chunks=30,
                        seed=0,
                    )

                    recording_hp_for_bad_detection = si.highpass_filter(
                        recording,
                        freq_min=300.0
                    )

                    bad_channel_ids, mad_bad_channel_report = detect_mad_bad_channels(
                        recording_for_detection=recording_hp_for_bad_detection,
                        label=f"{safe_id}_raw_highpass",
                        output_dir=diagnostics_dir,
                        seed=0,
                    )

                    if len(bad_channel_ids) > 0:
                        recording = recording.remove_channels(bad_channel_ids)

                    print(f"[{safe_id}] channels after raw cleanup: {len(recording.get_channel_ids())}")

                    # ==========================================
                    # Main preprocessing
                    # ==========================================

                    recording_f = si.bandpass_filter(
                        recording,
                        freq_min=300.0,
                        freq_max=3000.0
                    )

                    recording_preprocessed = si.common_reference(
                        recording_f,
                        reference="global",
                        operator="median"
                    )

                    recording_preprocessed = cast_to_float32(recording_preprocessed)

                    diagnose_recording_scaling(
                        recording_preprocessed,
                        label=f"{safe_id}_preprocessed_before_zero_noise_removal",
                        output_dir=diagnostics_dir,
                        seconds=2,
                    )

                    recording_preprocessed, preprocessed_zero_noise_report = remove_zero_noise_channels(
                        recording=recording_preprocessed,
                        label=f"{safe_id}_preprocessed",
                        output_dir=diagnostics_dir,
                        noise_floor_abs=1e-12,
                        noise_floor_rel=1e-6,
                        chunk_s=0.5,
                        num_chunks=30,
                        seed=1,
                    )

                    print(
                        f"[{safe_id}] channels after preprocessing cleanup: "
                        f"{len(recording_preprocessed.get_channel_ids())}"
                    )

                    # ==========================================
                    # Cache cleaned voltage traces
                    # ==========================================

                    cache_folder = os.path.join(kilosorter_output_dir, f"cache_{safe_id}")

                    if os.path.exists(cache_folder):
                        shutil.rmtree(cache_folder)

                    recording_cached = recording_preprocessed.save(
                        folder=cache_folder,
                        overwrite=True
                    )

                    diagnose_recording_scaling(
                        recording_cached,
                        label=f"{safe_id}_cached",
                        output_dir=diagnostics_dir,
                        seconds=2,
                    )

                    # ==========================================
                    # Run Kilosort4 fresh
                    # ==========================================

                    ks4_output_folder = os.path.join(
                        kilosorter_output_dir,
                        f"ks4_{safe_id}"
                    )

                    params = ss.get_default_sorter_params("kilosort4")
                    params["batch_size"] = 20000

                    # Make this explicit because the log showed Kilosort skipping drift correction.
                    params["nblocks"] = 0
                    params["do_correction"] = False

                    sorting = run_kilosort4_fresh(
                        recording_cached=recording_cached,
                        ks4_output_folder=ks4_output_folder,
                        safe_id=safe_id,
                        params=params,
                    )

                    if len(sorting.unit_ids) == 0:
                        print(f"[{safe_id}] No units found. Skipping.")

                        remove_folder_if_requested(
                            cache_folder,
                            enabled=DELETE_CACHE_WHEN_NO_UNITS,
                            label=f"{safe_id} cached preprocessed traces",
                        )

                        remove_folder_if_requested(
                            ks4_output_folder,
                            enabled=DELETE_KILOSORT_WHEN_NO_UNITS,
                            label=f"{safe_id} Kilosort4 output",
                        )

                        continue

                    print(f"[{safe_id}] Kilosort units found: {len(sorting.unit_ids)}")

                    # ==========================================
                    # SortingAnalyzer
                    # ==========================================

                    analyzer_folder = os.path.join(kilosorter_output_dir, f"analyzer_{safe_id}")

                    if os.path.exists(analyzer_folder):
                        shutil.rmtree(analyzer_folder)

                    analyzer = si.create_sorting_analyzer(
                        sorting=sorting,
                        recording=recording_cached,
                        format="binary_folder",
                        folder=analyzer_folder,
                        overwrite=True,
                    )

                    analyzer.compute("random_spikes", seed=0)
                    analyzer.compute("waveforms")
                    analyzer.compute("noise_levels")

                    analyzer_noise = analyzer.get_extension("noise_levels").get_data()
                    analyzer_channel_ids = np.asarray(analyzer.channel_ids)

                    analyzer_noise_invalid = (
                        (~np.isfinite(analyzer_noise)) |
                        (analyzer_noise <= 1e-12)
                    )

                    analyzer_noise_report = pd.DataFrame({
                        "channel_id": analyzer_channel_ids,
                        "noise_level": analyzer_noise,
                        "noise_invalid": analyzer_noise_invalid,
                    })

                    analyzer_noise_report.to_csv(
                        os.path.join(diagnostics_dir, f"{safe_id}_analyzer_noise_levels.csv"),
                        index=False
                    )

                    print(
                        f"[{safe_id}] analyzer invalid noise channels:",
                        int(np.sum(analyzer_noise_invalid))
                    )

                    analyzer.compute("templates")
                    
                    # ==========================================
                    # Waveform + spatial footprint features
                    # ==========================================

                    waveform_spatial_df = compute_waveform_spatial_features(
                        analyzer=analyzer,
                        recording=recording_cached,
                        unit_ids=sorting.unit_ids,
                        relative_threshold=FOOTPRINT_RELATIVE_THRESHOLD,
                        min_p2p_uv=FOOTPRINT_MIN_P2P_UV,
                    )


                    ### goes in phenotype matrix, not saving it anymore
                    # waveform_spatial_path = os.path.join(
                    #     diagnostics_dir,
                    #     f"{safe_id}_waveform_spatial_features.csv"
                    # )

                    # waveform_spatial_df.to_csv(
                    #     waveform_spatial_path,
                    #     index_label="unit_id",
                    # )

                    # print(f"[{safe_id}] Saved waveform/spatial features: {waveform_spatial_path}")

                    # ==========================================
                    # Unit locations — FIXED FOR 2D OR 3D OUTPUT
                    # ==========================================

                    unit_locations_array = analyzer.compute(
                        "unit_locations",
                        method="monopolar_triangulation"
                    ).get_data()

                    loc_df = make_unit_locations_df(
                        unit_locations_array=unit_locations_array,
                        unit_ids=sorting.unit_ids,
                        safe_id=safe_id,
                    )

                    # unit_locations_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_unit_locations.csv"
                    # )
                    # loc_df.to_csv(unit_locations_path, index_label="unit_id")

                    # ==========================================
                    # Quality metrics and finite-SNR filtering
                    # ==========================================

                    metrics = compute_quality_metrics_safe(
                        analyzer=analyzer,
                        safe_id=safe_id
                    )

                    num_invalid_snr = int(metrics["snr_invalid"].sum())
                    num_finite_snr = int(metrics["snr"].notna().sum())

                    print(f"[{safe_id}] finite SNR units: {num_finite_snr}")
                    print(f"[{safe_id}] invalid SNR units: {num_invalid_snr}")

                    qc_mask = (
                        metrics["snr"].notna() &
                        (metrics["snr"] >= 5) &
                        (metrics["firing_rate"] >= 0.005)
                    )

                    if "presence_ratio" in metrics.columns:
                        qc_mask = qc_mask & (metrics["presence_ratio"].fillna(0) >= 0.3)

                    if "isi_violation" in metrics.columns:
                        qc_mask = qc_mask & (metrics["isi_violation"].fillna(0) <= 0.5)

                    if "amplitude_cutoff" in metrics.columns:
                        qc_mask = qc_mask & (
                            metrics["amplitude_cutoff"].isna() |
                            (metrics["amplitude_cutoff"] <= 0.5)
                        )

                    metrics["qc_pass"] = qc_mask
                    good_unit_ids = metrics.index[qc_mask]

                    print(
                        f"[{safe_id}] Sorted {len(sorting.unit_ids)} total units. "
                        f"Kept {len(good_unit_ids)} finite-SNR high-quality units."
                    )

                    # metrics_filtered_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_quality_metrics_filtered.csv"
                    # )

                    # metrics.loc[good_unit_ids].to_csv(
                    #     metrics_filtered_path,
                    #     index_label="unit_id"
                    # )

                    if len(good_unit_ids) == 0:
                        print(f"[{safe_id}] No finite-SNR good units. Skipping phenotype export.")
                        continue

                    # ==========================================
                    # Convert SpikeInterface spikes
                    # ==========================================

                    spike_times_list = []
                    unit_ids_list = list(good_unit_ids)

                    for unit_id in unit_ids_list:
                        spike_times_s = sorting.get_unit_spike_train(unit_id) / sampling_rate
                        spike_times_list.append(np.asarray(spike_times_s, dtype=float))

                    # ==========================================
                    # Single-unit firing + burst features
                    # ==========================================

                    unit_feature_rows = []

                    for unit_id, spikes in zip(unit_ids_list, spike_times_list):
                        row = {"unit_id": unit_id}

                        # Firing pattern / ISI features.
                        row.update(
                            compute_single_unit_firing_features(
                                spike_times_s=spikes,
                                recording_duration_s=recording_duration_s,
                            )
                        )

                        # Keep your original fixed-ISI burst metrics.
                        row.update(
                            compute_burst_features(
                                spike_times_s=spikes,
                                recording_duration_s=recording_duration_s,
                                burst_isi_threshold_s=0.1,
                                min_spikes_per_burst=3,
                            )
                        )

                        # Add maximum-interval burst metrics.
                        row.update(
                            compute_max_interval_burst_features(
                                spike_times_s=spikes,
                                recording_duration_s=recording_duration_s,
                                params=MAX_INTERVAL_BURST_PARAMS,
                            )
                        )

                        unit_feature_rows.append(row)

                    unit_features_df = pd.DataFrame(unit_feature_rows).set_index("unit_id")

                    # unit_features_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_unit_firing_burst_features.csv",
                    # )

                    # unit_features_df.to_csv(
                    #     unit_features_path,
                    #     index_label="unit_id",
                    # )

                    # print(f"[{safe_id}] Saved firing/burst features: {unit_features_path}")

                    # ==========================================
                    # Perturbation-window features
                    # ==========================================

                    windowed_unit_df = compute_windowed_unit_features(
                        spike_times_list=spike_times_list,
                        unit_ids=unit_ids_list,
                        recording_duration_s=recording_duration_s,
                        windows=PERTURBATION_WINDOWS,
                    )

                    # windowed_unit_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_windowed_unit_features.csv",
                    # )

                    # windowed_unit_df.to_csv(windowed_unit_path, index=False)

                    windowed_unit_wide_df = make_windowed_unit_features_wide(
                        windowed_unit_df,
                        baseline_window="baseline_0_5min",
                    )

                    # windowed_unit_wide_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_windowed_unit_features_wide.csv",
                    # )

                    # windowed_unit_wide_df.to_csv(
                    #     windowed_unit_wide_path,
                    #     index_label="unit_id",
                    # )

                    # print(f"[{safe_id}] Saved windowed features: {windowed_unit_path}")

                    # ==========================================
                    # Network-burst features
                    # ==========================================

                    network_burst_summary, network_burst_events_df, population_rate_df = detect_network_bursts(
                        spike_times_list=spike_times_list,
                        unit_ids=unit_ids_list,
                        recording_duration_s=recording_duration_s,
                        bin_s=NETWORK_BURST_BIN_S,
                        smooth_bins=NETWORK_BURST_SMOOTH_BINS,
                        threshold_mad=NETWORK_BURST_THRESHOLD_MAD,
                        min_units_abs=NETWORK_BURST_MIN_UNITS_ABS,
                        min_units_frac=NETWORK_BURST_MIN_UNITS_FRAC,
                        min_duration_s=NETWORK_BURST_MIN_DURATION_S,
                        merge_gap_s=NETWORK_BURST_MERGE_GAP_S,
                    )

                    # network_events_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_network_burst_events.csv",
                    # )

                    # population_rate_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_population_rate_network_burst_trace.csv",
                    # )

                    # network_burst_events_df.to_csv(network_events_path, index=False)
                    # population_rate_df.to_csv(population_rate_path, index=False)

                    print(f"[{safe_id}] Network bursts detected: {network_burst_summary['network_n_bursts']}")

                    # ==========================================
                    # Surrogate-tested STTC connectivity
                    # ==========================================

                    (
                        sttc_matrix,
                        pvalue_matrix,
                        threshold_matrix,
                        significant_adjacency,
                        edge_df,
                        connectivity_included_ids,
                    ) = compute_surrogate_tested_sttc(
                        spike_times_list=spike_times_list,
                        unit_ids=unit_ids_list,
                        recording_duration_s=recording_duration_s,
                        dt_s=STTC_DT_S,
                        n_shuffles=STTC_N_SHUFFLES,
                        alpha=STTC_ALPHA,
                        min_spikes_per_unit=STTC_MIN_SPIKES_PER_UNIT,
                        max_units=CONNECTIVITY_MAX_UNITS,
                        seed=0,
                        progress_every_pairs=STTC_PROGRESS_EVERY_PAIRS,
                    )

                    # # Save compact edge table.
                    # edge_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_sttc_edges_surrogate_tested.csv",
                    # )

                    # edge_df.to_csv(edge_path, index=False)

                    # # Save significant edges only as compact output.
                    # significant_edge_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_sttc_significant_edges_only.csv",
                    # )

                    # if len(edge_df) > 0:
                    #     edge_df[edge_df["significant_edge"] == True].to_csv(
                    #         significant_edge_path,
                    #         index=False,
                    #     )
                    # else:
                    #     pd.DataFrame().to_csv(significant_edge_path, index=False)

                    # Optional dense matrix.
                    # Usually not needed for phenotype screening; edge tables are smaller.
                    if SAVE_DENSE_STTC_MATRIX:
                        sttc_matrix_path = os.path.join(
                            output_root,
                            f"{safe_id}_sttc_matrix.csv",
                        )

                        pd.DataFrame(
                            sttc_matrix,
                            index=connectivity_included_ids,
                            columns=connectivity_included_ids,
                        ).to_csv(sttc_matrix_path, index_label="unit_id")

                    # Per-unit mean STTC, aligned back to all good units.
                    mean_sttc_per_unit = pd.Series(np.nan, index=unit_ids_list, dtype=float)
                    connectivity_included_flag = pd.Series(False, index=unit_ids_list, dtype=bool)

                    if len(connectivity_included_ids) > 1:
                        mean_sttc_values = np.sum(sttc_matrix, axis=1) / (len(connectivity_included_ids) - 1)

                        for unit_id, value in zip(connectivity_included_ids, mean_sttc_values):
                            mean_sttc_per_unit.loc[unit_id] = value
                            connectivity_included_flag.loc[unit_id] = True

                    sttc_df = pd.DataFrame(
                        {
                            "mean_network_sttc": mean_sttc_per_unit,
                            "connectivity_included": connectivity_included_flag,
                            "sttc_dt_s": STTC_DT_S,
                            "sttc_n_shuffles": STTC_N_SHUFFLES,
                            "sttc_alpha": STTC_ALPHA,
                        },
                        index=unit_ids_list,
                    )

                    # ==========================================
                    # Graph features from significant STTC edges
                    # ==========================================

                    graph_well_features, unit_graph_df = compute_graph_features(
                        significant_adjacency=significant_adjacency,
                        sttc_matrix=sttc_matrix,
                        unit_ids=connectivity_included_ids,
                    )

                    # graph_well_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_graph_well_features.csv",
                    # )

                    # unit_graph_path = os.path.join(
                    #     OUTPUT_DIR,
                    #     f"{safe_id}_unit_graph_features.csv",
                    # )

                    # pd.DataFrame([graph_well_features]).to_csv(graph_well_path, index=False)
                    # unit_graph_df.to_csv(unit_graph_path, index_label="unit_id")

                    # print(f"[{safe_id}] Saved STTC and graph features.")

                    # ==========================================
                    # Build phenotype matrix
                    # ==========================================

                    well_df = metrics.loc[good_unit_ids].copy()

                    # Existing unit locations.
                    well_df = well_df.join(loc_df, how="left")

                    # New waveform/spatial footprint features.
                    well_df = well_df.join(waveform_spatial_df, how="left")

                    # New firing-pattern + burst metrics.
                    well_df = well_df.join(unit_features_df, how="left")

                    # New perturbation-window wide features.
                    well_df = well_df.join(windowed_unit_wide_df, how="left")

                    # New STTC metrics.
                    well_df = well_df.join(sttc_df, how="left")

                    # New graph unit-level metrics.
                    well_df = well_df.join(unit_graph_df, how="left")

                    # Add well-level network-burst features to every unit row.
                    for k, v in network_burst_summary.items():
                        well_df[k] = v

                    # Add well-level graph features to every unit row.
                    for k, v in graph_well_features.items():
                        well_df[k] = v

                    well_df["project_name"] = project_name
                    well_df["date_label"] = date_label
                    well_df["plate_label"] = plate_label
                    well_df["raw_h5_file"] = os.path.abspath(raw_file)
                    well_df["recording_id"] = TARGET_REC
                    well_df["well_id"] = well_name
                    well_df["safe_id"] = safe_id
                    #well_df["condition"] = CONDITION_LABEL
                    #well_df["celltype"] = CELLTYPE_LABEL
                    well_df["assay_number"] = assay_number
                    
                    csv_filename = os.path.join(
                        phenotype_dir,
                        f"{safe_id}_phenotype_features.csv"
                    )

                    well_df.to_csv(csv_filename, index_label="unit_id")

                    print(f"[{safe_id}] Saved well matrix: {csv_filename}")

                    # ==========================================
                    # Cleanup large intermediate folders
                    # ==========================================

                    remove_folder_if_requested(
                        cache_folder,
                        enabled=DELETE_CACHE_AFTER_SUCCESS,
                        label=f"{safe_id} cached preprocessed traces",
                    )

                    remove_folder_if_requested(
                        analyzer_folder,
                        enabled=DELETE_ANALYZER_AFTER_SUCCESS,
                        label=f"{safe_id} SortingAnalyzer folder",
                    )

                    remove_folder_if_requested(
                        ks4_output_folder,
                        enabled=DELETE_KILOSORT_OUTPUT_AFTER_SUCCESS,
                        label=f"{safe_id} Kilosort4 output",
                    )

                except Exception as e:
                    print(f"ERROR processing {safe_id}: {repr(e)}")
                    continue

        # ==========================================
        # 5. COMBINE ALL WELLS INTO MASTER MATRIX
        # ==========================================

        print("\nMerging individual well data into a unified master file...")

        csv_files = glob.glob(
            os.path.join(phenotype_dir, "*_phenotype_features.csv")
        )

        valid_dfs = []

        for f in csv_files:
            try:
                df = pd.read_csv(f)

                required_cols = {
                    "raw_file_id",
                    "raw_relative_path",
                    "well_id",
                    "recording_id",
                    "safe_id",
                    "condition",
                    "celltype",
                }

                if required_cols.issubset(set(df.columns)):
                    valid_dfs.append(df)
                else:
                    print(f"Skipping old/incompatible phenotype file: {f}")

            except Exception as e:
                print(f"Could not read {f}: {repr(e)}")

        if len(valid_dfs) == 0:
            print(
                "No valid phenotype feature CSV files were generated. "
                "Check the per-well error messages and QC reports."
            )
            return None

        master_df = pd.concat(valid_dfs, ignore_index=True)

        if "snr" in master_df.columns:
            master_df["snr"] = master_df["snr"].replace([np.inf, -np.inf], np.nan)


        if master_df is not None and len(master_df) > 0:
            all_master_dfs.append(master_df)


    phenotype_matrix = pd.concat(all_master_dfs, ignore_index=True)
    well_phenotype = make_well_summary(phenotype_matrix)

    phenotype_matrix.to_csv(output_root/ "phenotype_matrix.csv", index=False)
    well_phenotype.to_csv(output_root / "well_phenotype.csv", index=False)

    print(f"SUCCESS: Pipeline complete. Output matrices saved to {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", required=True,)
    parser.add_argument("--date_label", required=True)
    parser.add_argument("--plate_label", required=True)
    parser.add_argument( "--settings_path", default="utils/default_analysis_settings.json" )
    args = parser.parse_args()


    main(
        project_name=args.project_name,
        date_label=args.date_label,
        plate_label=args.plate_label,
        settings = args.settings_path
    )