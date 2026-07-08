import os
import sys
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
import quantities as pq
import neo
import h5py
import json
from pathlib import Path
import networkx as nx
import torch
from copy import deepcopy
torch.cuda.empty_cache()

#%% ############### PARAMETERS


def load_analysis_settings(settings_path: str | Path | None = None) -> dict:
    if settings_path is None:
        settings_path = Path(__file__).parent / "default_analysis_settings.json"
    else:
        settings_path = Path(settings_path)

    with settings_path.open("r") as f:
        return json.load(f)


#%%  ######### HELPERS


def remove_folder_if_requested(folder_path, enabled, label):
    """
    Safely remove a large intermediate folder after successful feature export.
    """
    if not enabled:
        return

    if folder_path is None:
        return

    if not os.path.exists(folder_path):
        return

    try:
        shutil.rmtree(folder_path)
        print(f"Deleted {label}: {folder_path}")
    except Exception as e:
        print(f"WARNING: Could not delete {label}: {folder_path}")
        print(f"Reason: {repr(e)}")

def cast_to_float32(recording):
    """
    Ensure filtering/noise estimation are done on float32 traces.
    """
    try:
        return si.astype(recording, dtype="float32")
    except Exception:
        return recording.astype("float32")


def get_num_frames_safe(recording, segment_index=0):
    """
    Compatible frame-count getter across SpikeInterface versions.
    """
    try:
        return recording.get_num_frames(segment_index=segment_index)
    except Exception:
        return recording.get_num_samples(segment_index=segment_index)


def get_traces_safe(recording, start_frame, end_frame, return_in_uV=True):
    """
    Compatible trace getter across SpikeInterface versions.
    """
    try:
        return recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            return_in_uV=return_in_uV
        )
    except TypeError:
        try:
            return recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame,
                return_scaled=return_in_uV
            )
        except TypeError:
            return recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame
            )

#%% ######### DIAGNOSTICS


def orient_traces_samples_by_channels(traces, n_channels):
    """
    Ensure traces are samples x channels.
    """
    traces = np.asarray(traces)

    if traces.ndim != 2:
        raise ValueError(f"Expected 2D traces, got shape {traces.shape}")

    if traces.shape[1] == n_channels:
        return traces

    if traces.shape[0] == n_channels:
        return traces.T

    return traces


def unsigned_to_signed_safe(recording, safe_id, bit_depth=10):
    """
    Apply unsigned_to_signed only for unsigned recordings.

    For this MaxWell file, the raw uint16 values are centered around ~512,
    so bit_depth=10 is appropriate unless metadata proves otherwise.
    """
    try:
        dtype = recording.get_dtype()
    except Exception:
        dtype = None

    if dtype is not None and np.issubdtype(dtype, np.unsignedinteger):
        print(f"[{safe_id}] unsigned dtype detected: {dtype}")
        print(f"[{safe_id}] applying unsigned_to_signed(bit_depth={bit_depth})")

        try:
            return si.unsigned_to_signed(recording, bit_depth=bit_depth)
        except TypeError:
            print(
                f"[{safe_id}] WARNING: this SpikeInterface version may not support "
                f"bit_depth. Falling back to unsigned_to_signed(recording)."
            )
            return si.unsigned_to_signed(recording)

    print(f"[{safe_id}] recording not unsigned; skipping unsigned_to_signed()")
    return recording


def diagnose_recording_scaling(recording, label, output_dir, seconds=2):
    """
    Diagnose raw-vs-scaled traces, gain metadata, flat channels, and clipping.
    """
    fs = recording.get_sampling_frequency()
    n_channels = len(recording.get_channel_ids())
    n_frames_total = get_num_frames_safe(recording)
    n_frames = int(min(seconds * fs, n_frames_total))

    channel_ids = np.asarray(recording.get_channel_ids())

    print(f"\n[{label}] ===== RECORDING / SCALING DIAGNOSTIC =====")

    try:
        print(f"[{label}] dtype:", recording.get_dtype())
    except Exception:
        print(f"[{label}] dtype: unavailable")

    print(f"[{label}] channels:", n_channels)
    print(f"[{label}] sampling rate:", fs)
    print(f"[{label}] duration:", recording.get_total_duration())

    try:
        gains = recording.get_property("gain_to_uV")
    except Exception:
        gains = None

    try:
        offsets = recording.get_property("offset_to_uV")
    except Exception:
        offsets = None

    print(f"[{label}] gain_to_uV exists:", gains is not None)
    print(f"[{label}] offset_to_uV exists:", offsets is not None)

    if gains is not None:
        gains = np.asarray(gains, dtype=float)
        print(
            f"[{label}] gain_to_uV min/median/max:",
            np.nanmin(gains),
            np.nanmedian(gains),
            np.nanmax(gains)
        )
        print(f"[{label}] zero gains:", int(np.sum(gains == 0)))
        print(f"[{label}] non-finite gains:", int(np.sum(~np.isfinite(gains))))

    if offsets is not None:
        offsets = np.asarray(offsets, dtype=float)
        print(
            f"[{label}] offset_to_uV min/median/max:",
            np.nanmin(offsets),
            np.nanmedian(offsets),
            np.nanmax(offsets)
        )
        print(f"[{label}] non-finite offsets:", int(np.sum(~np.isfinite(offsets))))

    raw = get_traces_safe(
        recording,
        start_frame=0,
        end_frame=n_frames,
        return_in_uV=False
    )

    raw = orient_traces_samples_by_channels(raw, n_channels)
    raw = np.asarray(raw)

    raw_std = np.nanstd(raw, axis=0)
    raw_min_ch = np.nanmin(raw, axis=0)
    raw_max_ch = np.nanmax(raw, axis=0)

    print(
        f"[{label}] raw global min/max/std:",
        np.nanmin(raw),
        np.nanmax(raw),
        np.nanstd(raw)
    )

    print(f"[{label}] raw zero-std channels:", int(np.sum(raw_std == 0)))
    print(f"[{label}] raw near-zero-std channels:", int(np.sum(raw_std < 1e-12)))

    global_min = np.nanmin(raw)
    global_max = np.nanmax(raw)

    frac_at_min = np.mean(raw == global_min, axis=0)
    frac_at_max = np.mean(raw == global_max, axis=0)

    possible_clipped = (frac_at_min > 0.001) | (frac_at_max > 0.001)

    print(f"[{label}] possible clipped channels:", int(np.sum(possible_clipped)))
    print(f"[{label}] max fraction at global min:", float(np.nanmax(frac_at_min)))
    print(f"[{label}] max fraction at global max:", float(np.nanmax(frac_at_max)))

    try:
        scaled = get_traces_safe(
            recording,
            start_frame=0,
            end_frame=n_frames,
            return_in_uV=True
        )

        scaled = orient_traces_samples_by_channels(scaled, n_channels)
        scaled = np.asarray(scaled, dtype=np.float32)

        scaled_std = np.nanstd(scaled, axis=0)

        print(
            f"[{label}] scaled uV global min/max/std:",
            np.nanmin(scaled),
            np.nanmax(scaled),
            np.nanstd(scaled)
        )
        print(f"[{label}] scaled zero-std channels:", int(np.sum(scaled_std == 0)))
        print(f"[{label}] scaled near-zero-std channels:", int(np.sum(scaled_std < 1e-12)))

    except Exception as e:
        print(f"[{label}] scaled trace unavailable:", repr(e))
        scaled_std = np.full(n_channels, np.nan)

    report = pd.DataFrame({
        "channel_id": channel_ids,
        "raw_std": raw_std,
        "raw_min": raw_min_ch,
        "raw_max": raw_max_ch,
        "frac_at_global_min": frac_at_min,
        "frac_at_global_max": frac_at_max,
        "possible_clipped": possible_clipped,
        "scaled_std_uV": scaled_std,
    })

    if gains is not None and len(gains) == len(report):
        report["gain_to_uV"] = gains

    if offsets is not None and len(offsets) == len(report):
        report["offset_to_uV"] = offsets

    report_path = os.path.join(
        output_dir,
        f"{label}_recording_scaling_diagnostic.csv"
    )

    report.to_csv(report_path, index=False)

    return report


def estimate_channel_noise_mad(
    recording,
    chunk_s=0.5,
    num_chunks=30,
    seed=0,
    return_in_uV=True,
):
    """
    Estimate per-channel noise using MAD across random chunks.
    """
    rng = np.random.default_rng(seed)

    fs = recording.get_sampling_frequency()
    n_frames = get_num_frames_safe(recording)
    n_channels = len(recording.get_channel_ids())
    channel_ids = np.asarray(recording.get_channel_ids())

    chunk_size = int(chunk_s * fs)
    chunk_size = max(1, min(chunk_size, n_frames))

    if n_frames <= chunk_size:
        starts = np.array([0])
    else:
        starts = rng.integers(
            low=0,
            high=n_frames - chunk_size,
            size=num_chunks
        )

    noise_chunks = []

    for start in starts:
        start = int(start)
        end = int(start + chunk_size)

        traces = get_traces_safe(
            recording,
            start_frame=start,
            end_frame=end,
            return_in_uV=return_in_uV
        )

        traces = orient_traces_samples_by_channels(traces, n_channels)
        traces = np.asarray(traces, dtype=np.float32)

        med = np.nanmedian(traces, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(traces - med), axis=0)

        noise_chunks.append(mad)

    noise = np.nanmedian(np.vstack(noise_chunks), axis=0)

    return channel_ids, noise


def remove_zero_noise_channels(
    recording,
    label,
    output_dir,
    noise_floor_abs=1e-12,
    noise_floor_rel=1e-6,
    chunk_s=0.5,
    num_chunks=30,
    seed=0,
):
    """
    Remove zero / near-zero / non-finite noise channels.
    """
    channel_ids, noise = estimate_channel_noise_mad(
        recording,
        chunk_s=chunk_s,
        num_chunks=num_chunks,
        seed=seed,
        return_in_uV=True,
    )

    finite_positive = noise[np.isfinite(noise) & (noise > 0)]

    if len(finite_positive) == 0:
        adaptive_floor = noise_floor_abs
        print(f"[{label}] WARNING: no finite positive channel noise found.")
    else:
        adaptive_floor = max(
            noise_floor_abs,
            noise_floor_rel * np.nanmedian(finite_positive)
        )

    bad_mask = (~np.isfinite(noise)) | (noise <= adaptive_floor)
    bad_channel_ids = channel_ids[bad_mask]

    report = pd.DataFrame({
        "channel_id": channel_ids,
        "noise_mad_uV": noise,
        "adaptive_noise_floor": adaptive_floor,
        "removed_zero_or_flat": bad_mask,
    })

    report_path = os.path.join(
        output_dir,
        f"{label}_zero_noise_channel_report.csv"
    )

    report.to_csv(report_path, index=False)

    print(f"[{label}] channels before zero-noise removal: {len(channel_ids)}")
    print(f"[{label}] adaptive noise floor: {adaptive_floor}")
    print(f"[{label}] zero/flat channels removed: {len(bad_channel_ids)}")

    if len(bad_channel_ids) > 0:
        recording = recording.remove_channels(list(bad_channel_ids))

    return recording, report


def detect_mad_bad_channels(recording_for_detection, label, output_dir, seed=0):
    """
    Detect noisy/abnormal channels using SpikeInterface's MAD-based detector.
    """
    try:
        bad_channel_ids, channel_labels = si.detect_bad_channels(
            recording_for_detection,
            method="mad",
            std_mad_threshold=5,
            seed=seed,
        )
    except TypeError:
        bad_channel_ids, channel_labels = si.detect_bad_channels(
            recording_for_detection,
            method="mad",
        )

    channel_ids = np.asarray(recording_for_detection.get_channel_ids())

    report = pd.DataFrame({
        "channel_id": channel_ids,
        "bad_channel_label": channel_labels,
        "removed_by_detect_bad_channels": np.isin(channel_ids, bad_channel_ids),
    })

    report_path = os.path.join(
        output_dir,
        f"{label}_mad_bad_channel_report.csv"
    )

    report.to_csv(report_path, index=False)

    print(f"[{label}] MAD bad channels detected: {len(bad_channel_ids)}")

    return list(bad_channel_ids), report


#%% ####### ANALYSIS HELPERS

def make_unit_locations_df(unit_locations_array, unit_ids, safe_id):
    """
    Safely convert unit_locations output to a dataframe.

    Some SpikeInterface methods return:
      - 2 columns: x, y
      - 3 columns: x, y, z
      - occasionally other dimensional outputs

    """
    arr = np.asarray(unit_locations_array)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    unit_ids = list(unit_ids)

    if arr.shape[0] != len(unit_ids) and arr.shape[1] == len(unit_ids):
        print(f"[{safe_id}] Transposing unit_locations array from {arr.shape}")
        arr = arr.T

    if arr.shape[0] != len(unit_ids):
        raise ValueError(
            f"[{safe_id}] unit_locations shape {arr.shape} does not match "
            f"number of units {len(unit_ids)}"
        )

    print(f"[{safe_id}] unit_locations shape:", arr.shape)

    if arr.shape[1] == 2:
        columns = ["pos_x_um", "pos_y_um"]

    elif arr.shape[1] == 3:
        columns = ["pos_x_um", "pos_y_um", "pos_z_um"]

    else:
        columns = [f"unit_location_dim_{i}" for i in range(arr.shape[1])]

        if arr.shape[1] >= 1:
            columns[0] = "pos_x_um"
        if arr.shape[1] >= 2:
            columns[1] = "pos_y_um"
        if arr.shape[1] >= 3:
            columns[2] = "pos_z_um"

    loc_df = pd.DataFrame(
        arr,
        index=unit_ids,
        columns=columns
    )

    return loc_df


def compute_burst_features(
    spike_times_s,
    recording_duration_s,
    burst_isi_threshold_s=0.1,
    min_spikes_per_burst=3,
):
    """
    Burst-like grouping based on consecutive short ISIs.
    """
    spikes = np.asarray(spike_times_s, dtype=float)

    if len(spikes) < 3:
        return {
            "burst_rate_hz": 0.0,
            "n_bursts": 0,
            "mean_burst_duration_s": 0.0,
            "mean_short_isi_s": 0.0,
            "cv_isi": 0.0,
        }

    isi = np.diff(spikes)

    cv_isi = np.nanstd(isi) / (np.nanmean(isi) + 1e-12)

    short_mask = isi < burst_isi_threshold_s

    if not np.any(short_mask):
        return {
            "burst_rate_hz": 0.0,
            "n_bursts": 0,
            "mean_burst_duration_s": 0.0,
            "mean_short_isi_s": 0.0,
            "cv_isi": cv_isi,
        }

    burst_durations = []
    mean_short_isis = []

    i = 0

    while i < len(short_mask):
        if not short_mask[i]:
            i += 1
            continue

        start_isi = i

        while i < len(short_mask) and short_mask[i]:
            i += 1

        end_isi = i - 1

        start_spike = start_isi
        end_spike = end_isi + 1

        n_spikes_in_burst = end_spike - start_spike + 1

        if n_spikes_in_burst >= min_spikes_per_burst:
            duration = spikes[end_spike] - spikes[start_spike]
            burst_durations.append(duration)
            mean_short_isis.append(np.mean(isi[start_isi:end_isi + 1]))

    n_bursts = len(burst_durations)

    if n_bursts == 0:
        burst_rate_hz = 0.0
        mean_burst_duration = 0.0
        mean_short_isi = 0.0
    else:
        burst_rate_hz = n_bursts / max(recording_duration_s, 1e-12)
        mean_burst_duration = float(np.mean(burst_durations))
        mean_short_isi = float(np.mean(mean_short_isis))

    return {
        "burst_rate_hz": burst_rate_hz,
        "n_bursts": n_bursts,
        "mean_burst_duration_s": mean_burst_duration,
        "mean_short_isi_s": mean_short_isi,
        "cv_isi": cv_isi,
    }


def fraction_time_covered_by_spikes(spike_times_s, dt_s, t_start_s, t_stop_s):
    """
    Fraction of recording time covered by union of intervals:
    [spike - dt, spike + dt].
    """
    spikes = np.asarray(spike_times_s, dtype=float)

    if len(spikes) == 0:
        return 0.0

    intervals = np.column_stack([
        np.maximum(spikes - dt_s, t_start_s),
        np.minimum(spikes + dt_s, t_stop_s),
    ])

    intervals = intervals[intervals[:, 1] > intervals[:, 0]]

    if len(intervals) == 0:
        return 0.0

    intervals = intervals[np.argsort(intervals[:, 0])]

    merged = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))

    covered_time = sum(end - start for start, end in merged)
    total_time = max(t_stop_s - t_start_s, 1e-12)

    return covered_time / total_time


def sttc_correct(ts1, ts2, dt_s, t_start_s, t_stop_s):
    """
    Spike Time Tiling Coefficient.
    """
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)

    if len(ts1) < 2 or len(ts2) < 2:
        return 0.0

    PA = np.mean([np.any(np.abs(ts2 - t) <= dt_s) for t in ts1])
    PB = np.mean([np.any(np.abs(ts1 - t) <= dt_s) for t in ts2])

    TA = fraction_time_covered_by_spikes(ts1, dt_s, t_start_s, t_stop_s)
    TB = fraction_time_covered_by_spikes(ts2, dt_s, t_start_s, t_stop_s)

    denom_a = 1.0 - PA * TB
    denom_b = 1.0 - PB * TA

    term_a = (PA - TB) / denom_a if abs(denom_a) > 1e-12 else 0.0
    term_b = (PB - TA) / denom_b if abs(denom_b) > 1e-12 else 0.0

    return 0.5 * (term_a + term_b)


def run_kilosort4_fresh(
    recording_cached,
    ks4_output_folder,
    safe_id,
    params,
):
    """
    Always rerun Kilosort4 from scratch.

    Any existing Kilosort4 output folder is deleted first, so the pipeline never
    loads old sorter output from a previous run.
    """
    if os.path.exists(ks4_output_folder):
        print(f"[{safe_id}] Removing existing Kilosort4 folder before fresh run:")
        print(f"[{safe_id}] {ks4_output_folder}")
        shutil.rmtree(ks4_output_folder)

    sorting = ss.run_sorter(
        sorter_name="kilosort4",
        recording=recording_cached,
        folder=ks4_output_folder,
        verbose=True,
        docker_image=False,
        remove_existing_folder=True,
        **params
    )

    return sorting


#%% # ==========================================

def get_channel_locations_dataframe(recording, channel_ids=None):
    """
    Return channel locations as a dataframe indexed by channel_id.
    """
    if channel_ids is None:
        channel_ids = np.asarray(recording.get_channel_ids())
    else:
        channel_ids = np.asarray(channel_ids)

    try:
        locs = recording.get_channel_locations()
        locs = np.asarray(locs, dtype=float)
    except Exception:
        locs = np.full((len(channel_ids), 2), np.nan)

    if locs.ndim == 1:
        locs = locs.reshape(-1, 1)

    # Align length defensively.
    if locs.shape[0] != len(channel_ids):
        locs = np.full((len(channel_ids), 2), np.nan)

    columns = []
    if locs.shape[1] >= 1:
        columns.append("channel_x_um")
    if locs.shape[1] >= 2:
        columns.append("channel_y_um")
    if locs.shape[1] >= 3:
        columns.append("channel_z_um")

    while len(columns) < locs.shape[1]:
        columns.append(f"channel_location_dim_{len(columns)}")

    return pd.DataFrame(locs, index=channel_ids, columns=columns)


def get_templates_and_unit_ids_safe(analyzer, fallback_unit_ids):
    """
    Safely get templates and unit IDs from SortingAnalyzer.

    Expected template shape is:
        n_units x n_samples x n_channels
    """
    fallback_unit_ids = list(fallback_unit_ids)

    try:
        templates = analyzer.get_extension("templates").get_data()
    except Exception:
        templates = analyzer.compute("templates").get_data()

    templates = np.asarray(templates)

    try:
        template_unit_ids = list(analyzer.unit_ids)
    except Exception:
        template_unit_ids = fallback_unit_ids

    if templates.ndim != 3:
        raise ValueError(f"Expected 3D templates, got shape {templates.shape}")

    if templates.shape[0] != len(template_unit_ids):
        print(
            "WARNING: template unit dimension does not match analyzer.unit_ids. "
            "Falling back to supplied unit IDs."
        )
        template_unit_ids = fallback_unit_ids

    return templates, template_unit_ids


def compute_waveform_spatial_features(
    analyzer,
    recording,
    unit_ids,
    relative_threshold=0.25,
    min_p2p_uv=10.0,
):
    """
    Compute waveform and spatial-footprint features.

    Output columns:
        template_p2p_max_uV
        template_best_channel_id
        template_best_channel_x_um
        template_best_channel_y_um
        template_trough_to_peak_ms
        template_half_width_ms
        template_repolarization_slope_uV_per_ms
        template_footprint_n_channels
        template_footprint_radius_um
        template_footprint_area_um2
        template_footprint_weighted_centroid_x_um
        template_footprint_weighted_centroid_y_um
    """
    requested_unit_ids = list(unit_ids)
    channel_ids = np.asarray(analyzer.channel_ids)
    channel_locations = get_channel_locations_dataframe(
        recording,
        channel_ids=channel_ids,
    )

    try:
        templates, template_unit_ids = get_templates_and_unit_ids_safe(
            analyzer,
            fallback_unit_ids=requested_unit_ids,
        )
    except Exception as e:
        print(f"WARNING: Could not get templates for waveform features: {repr(e)}")
        return pd.DataFrame(index=requested_unit_ids)

    fs = float(analyzer.sampling_frequency)
    rows = []

    for unit_index, unit_id in enumerate(template_unit_ids):
        if unit_id not in requested_unit_ids:
            continue

        template = np.asarray(templates[unit_index], dtype=float)

        # Make sure template is samples x channels.
        if template.shape[1] != len(channel_ids) and template.shape[0] == len(channel_ids):
            template = template.T

        if template.ndim != 2 or template.shape[1] != len(channel_ids):
            rows.append({"unit_id": unit_id})
            continue

        p2p = np.nanmax(template, axis=0) - np.nanmin(template, axis=0)

        if np.all(~np.isfinite(p2p)):
            rows.append({"unit_id": unit_id})
            continue

        best_ch_idx = int(np.nanargmax(p2p))
        best_channel_id = channel_ids[best_ch_idx]
        max_p2p = float(p2p[best_ch_idx])

        best_waveform = template[:, best_ch_idx]
        trough_idx = int(np.nanargmin(best_waveform))
        trough_val = float(best_waveform[trough_idx])

        # Peak after trough.
        if trough_idx < len(best_waveform) - 1:
            peak_after_idx = trough_idx + int(np.nanargmax(best_waveform[trough_idx:]))
            peak_after_val = float(best_waveform[peak_after_idx])
            trough_to_peak_ms = 1000.0 * (peak_after_idx - trough_idx) / fs
            repolarization_slope = (
                (peak_after_val - trough_val) / max(trough_to_peak_ms, 1e-12)
            )
        else:
            peak_after_idx = np.nan
            trough_to_peak_ms = np.nan
            repolarization_slope = np.nan

        # Half-width around negative trough.
        half_width_ms = np.nan
        try:
            half_level = trough_val / 2.0
            below_half = np.where(best_waveform <= half_level)[0]
            if len(below_half) >= 2:
                half_width_ms = 1000.0 * (below_half[-1] - below_half[0]) / fs
        except Exception:
            half_width_ms = np.nan

        # Spatial footprint from channels with large p2p.
        footprint_threshold = max(relative_threshold * max_p2p, min_p2p_uv)
        footprint_mask = p2p >= footprint_threshold
        footprint_indices = np.where(footprint_mask)[0]

        best_x = np.nan
        best_y = np.nan

        if "channel_x_um" in channel_locations.columns:
            best_x = channel_locations.iloc[best_ch_idx]["channel_x_um"]
        if "channel_y_um" in channel_locations.columns:
            best_y = channel_locations.iloc[best_ch_idx]["channel_y_um"]

        footprint_radius = np.nan
        footprint_area = np.nan
        weighted_centroid_x = np.nan
        weighted_centroid_y = np.nan

        if (
            len(footprint_indices) > 0
            and "channel_x_um" in channel_locations.columns
            and "channel_y_um" in channel_locations.columns
        ):
            xy = channel_locations.iloc[footprint_indices][
                ["channel_x_um", "channel_y_um"]
            ].to_numpy(dtype=float)

            weights = p2p[footprint_indices].astype(float)

            valid = (
                np.all(np.isfinite(xy), axis=1)
                & np.isfinite(weights)
                & (weights > 0)
            )

            if np.any(valid):
                xy_valid = xy[valid]
                weights_valid = weights[valid]

                centroid = np.average(xy_valid, axis=0, weights=weights_valid)
                weighted_centroid_x = float(centroid[0])
                weighted_centroid_y = float(centroid[1])

                distances = np.sqrt(np.sum((xy_valid - centroid) ** 2, axis=1))
                footprint_radius = float(np.nanmax(distances)) if len(distances) else np.nan

                if ConvexHull is not None and len(xy_valid) >= 3:
                    try:
                        hull = ConvexHull(xy_valid)
                        footprint_area = float(hull.volume)
                    except Exception:
                        footprint_area = np.nan

        rows.append({
            "unit_id": unit_id,
            "template_p2p_max_uV": max_p2p,
            "template_best_channel_id": best_channel_id,
            "template_best_channel_x_um": best_x,
            "template_best_channel_y_um": best_y,
            "template_trough_to_peak_ms": trough_to_peak_ms,
            "template_half_width_ms": half_width_ms,
            "template_repolarization_slope_uV_per_ms": repolarization_slope,
            "template_footprint_n_channels": int(np.sum(footprint_mask)),
            "template_footprint_radius_um": footprint_radius,
            "template_footprint_area_um2": footprint_area,
            "template_footprint_weighted_centroid_x_um": weighted_centroid_x,
            "template_footprint_weighted_centroid_y_um": weighted_centroid_y,
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return pd.DataFrame(index=requested_unit_ids)

    out = out.set_index("unit_id")
    out = out.reindex(requested_unit_ids)

    return out


def compute_single_unit_firing_features(
    spike_times_s,
    recording_duration_s,
):
    """
    Compute firing-pattern features for subtype classification.
    """
    spikes = np.asarray(spike_times_s, dtype=float)
    spikes = spikes[np.isfinite(spikes)]
    spikes = spikes[(spikes >= 0.0) & (spikes <= recording_duration_s)]

    n_spikes = len(spikes)
    firing_rate_from_spikes_hz = n_spikes / max(recording_duration_s, 1e-12)

    if n_spikes < 2:
        return {
            "n_spikes": n_spikes,
            "firing_rate_from_spikes_hz": firing_rate_from_spikes_hz,
            "mean_isi_s": np.nan,
            "median_isi_s": np.nan,
            "std_isi_s": np.nan,
            "cv_isi": np.nan,
            "lv_isi": np.nan,
            "isi_entropy": np.nan,
            "refractory_violation_fraction_2ms": np.nan,
            "silent_period_fraction_1s": np.nan,
        }

    isi = np.diff(spikes)

    mean_isi = float(np.nanmean(isi))
    median_isi = float(np.nanmedian(isi))
    std_isi = float(np.nanstd(isi))
    cv_isi = std_isi / (mean_isi + 1e-12)

    if len(isi) >= 2:
        lv_terms = 3.0 * ((isi[1:] - isi[:-1]) ** 2) / ((isi[1:] + isi[:-1] + 1e-12) ** 2)
        lv_isi = float(np.nanmean(lv_terms))
    else:
        lv_isi = np.nan

    # ISI entropy using log-spaced bins.
    isi_entropy = np.nan
    try:
        positive_isi = isi[isi > 0]
        if len(positive_isi) >= 5:
            bins = np.logspace(
                np.log10(max(np.nanmin(positive_isi), 1e-4)),
                np.log10(max(np.nanmax(positive_isi), 1e-3)),
                20,
            )
            hist, _ = np.histogram(positive_isi, bins=bins)
            p = hist / max(np.sum(hist), 1)
            p = p[p > 0]
            isi_entropy = float(-np.sum(p * np.log2(p)))
    except Exception:
        isi_entropy = np.nan

    refractory_violation_fraction = float(np.mean(isi < 0.002)) if len(isi) else np.nan
    silent_period_fraction_1s = float(np.mean(isi > 1.0)) if len(isi) else np.nan

    return {
        "n_spikes": n_spikes,
        "firing_rate_from_spikes_hz": firing_rate_from_spikes_hz,
        "mean_isi_s": mean_isi,
        "median_isi_s": median_isi,
        "std_isi_s": std_isi,
        "cv_isi": cv_isi,
        "lv_isi": lv_isi,
        "isi_entropy": isi_entropy,
        "refractory_violation_fraction_2ms": refractory_violation_fraction,
        "silent_period_fraction_1s": silent_period_fraction_1s,
    }


def compute_max_interval_burst_features(
    spike_times_s,
    recording_duration_s,
    params=None,
):
    """
    Maximum-interval burst detection.

    This adds richer burst metrics than the simple fixed-ISI detector.
    """
    if params is None:
        params = MAX_INTERVAL_BURST_PARAMS

    spikes = np.asarray(spike_times_s, dtype=float)
    spikes = spikes[np.isfinite(spikes)]
    spikes = spikes[(spikes >= 0.0) & (spikes <= recording_duration_s)]

    min_spikes = int(params.get("min_spikes_per_burst", 3))

    empty = {
        "burst_rate_hz_max_interval": 0.0,
        "n_bursts_max_interval": 0,
        "mean_burst_duration_s_max_interval": 0.0,
        "median_burst_duration_s_max_interval": 0.0,
        "mean_spikes_per_burst_max_interval": 0.0,
        "median_spikes_per_burst_max_interval": 0.0,
        "fraction_spikes_in_bursts_max_interval": 0.0,
        "mean_interburst_interval_s_max_interval": np.nan,
        "burst_duration_cv_max_interval": np.nan,
    }

    if len(spikes) < min_spikes:
        return empty

    max_start_isi = float(params.get("max_start_isi_s", 0.10))
    max_end_isi = float(params.get("max_end_isi_s", 0.20))
    min_interburst_interval = float(params.get("min_interburst_interval_s", 0.20))
    min_burst_duration = float(params.get("min_burst_duration_s", 0.005))

    isi = np.diff(spikes)
    candidate_bursts = []

    i = 0
    while i < len(isi):
        if isi[i] <= max_start_isi:
            start_spike = i

            j = i
            while j < len(isi) and isi[j] <= max_end_isi:
                j += 1

            end_spike = j
            n_spikes_burst = end_spike - start_spike + 1
            duration = spikes[end_spike] - spikes[start_spike]

            if n_spikes_burst >= min_spikes and duration >= min_burst_duration:
                candidate_bursts.append([start_spike, end_spike])

            i = max(j, i + 1)
        else:
            i += 1

    # Merge close bursts.
    merged = []
    for burst in candidate_bursts:
        if not merged:
            merged.append(burst)
            continue

        previous = merged[-1]
        gap = spikes[burst[0]] - spikes[previous[1]]

        if gap <= min_interburst_interval:
            previous[1] = burst[1]
        else:
            merged.append(burst)

    burst_durations = []
    burst_spike_counts = []
    burst_spike_indices = set()
    burst_starts = []

    for start_idx, end_idx in merged:
        n_spikes_burst = end_idx - start_idx + 1
        duration = spikes[end_idx] - spikes[start_idx]

        if n_spikes_burst >= min_spikes and duration >= min_burst_duration:
            burst_durations.append(duration)
            burst_spike_counts.append(n_spikes_burst)
            burst_starts.append(spikes[start_idx])

            for idx in range(start_idx, end_idx + 1):
                burst_spike_indices.add(idx)

    n_bursts = len(burst_durations)

    if n_bursts == 0:
        return empty

    interburst_intervals = np.diff(burst_starts) if len(burst_starts) > 1 else np.array([])

    return {
        "burst_rate_hz_max_interval": n_bursts / max(recording_duration_s, 1e-12),
        "n_bursts_max_interval": n_bursts,
        "mean_burst_duration_s_max_interval": float(np.mean(burst_durations)),
        "median_burst_duration_s_max_interval": float(np.median(burst_durations)),
        "mean_spikes_per_burst_max_interval": float(np.mean(burst_spike_counts)),
        "median_spikes_per_burst_max_interval": float(np.median(burst_spike_counts)),
        "fraction_spikes_in_bursts_max_interval": len(burst_spike_indices) / max(len(spikes), 1),
        "mean_interburst_interval_s_max_interval": (
            float(np.mean(interburst_intervals)) if len(interburst_intervals) else np.nan
        ),
        "burst_duration_cv_max_interval": (
            float(np.std(burst_durations) / (np.mean(burst_durations) + 1e-12))
            if len(burst_durations) > 1 else 0.0
        ),
    }


def smooth_population_counts(counts, smooth_bins=5):
    """
    Simple moving-average smoother for population spike counts.
    """
    counts = np.asarray(counts, dtype=float)

    if smooth_bins is None or smooth_bins <= 1:
        return counts

    kernel = np.ones(int(smooth_bins), dtype=float)
    kernel = kernel / np.sum(kernel)

    return np.convolve(counts, kernel, mode="same")


def detect_network_bursts(
    spike_times_list,
    unit_ids,
    recording_duration_s,
    bin_s=0.020,
    smooth_bins=5,
    threshold_mad=5.0,
    min_units_abs=3,
    min_units_frac=0.10,
    min_duration_s=0.040,
    merge_gap_s=0.100,
):
    """
    Detect well-level network bursts from population activity.

    Returns:
        network_burst_summary
        network_burst_events_df
        population_rate_df
    """
    unit_ids = list(unit_ids)
    n_units = len(unit_ids)

    empty_summary = {
        "network_n_bursts": 0,
        "network_burst_rate_hz": 0.0,
        "network_mean_burst_duration_s": 0.0,
        "network_median_burst_duration_s": 0.0,
        "network_mean_participating_units": 0.0,
        "network_mean_participation_fraction": 0.0,
        "network_mean_peak_population_rate_hz": 0.0,
        "network_threshold_population_count": np.nan,
    }

    if n_units == 0 or recording_duration_s <= 0:
        return empty_summary, pd.DataFrame(), pd.DataFrame()

    n_bins = int(np.ceil(recording_duration_s / bin_s))
    bin_edges = np.arange(n_bins + 1, dtype=float) * bin_s
    bin_edges[-1] = max(bin_edges[-1], recording_duration_s)

    population_counts = np.zeros(n_bins, dtype=float)
    unit_active_bin_sets = []

    for spikes in spike_times_list:
        spikes = np.asarray(spikes, dtype=float)
        spikes = spikes[np.isfinite(spikes)]
        spikes = spikes[(spikes >= 0.0) & (spikes < recording_duration_s)]

        counts, _ = np.histogram(spikes, bins=bin_edges)
        population_counts += counts
        unit_active_bin_sets.append(set(np.where(counts > 0)[0].tolist()))

    smoothed = smooth_population_counts(population_counts, smooth_bins=smooth_bins)

    baseline = np.nanmedian(smoothed)
    mad = 1.4826 * np.nanmedian(np.abs(smoothed - baseline))
    threshold = baseline + threshold_mad * mad

    if not np.isfinite(threshold) or threshold <= 0:
        threshold = max(1.0, np.nanpercentile(smoothed, 95))

    above = smoothed >= threshold

    segments = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue

        start_bin = i
        while i < len(above) and above[i]:
            i += 1
        end_bin = i - 1

        segments.append([start_bin, end_bin])

    # Merge nearby segments.
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        gap_s = (seg[0] - prev[1] - 1) * bin_s

        if gap_s <= merge_gap_s:
            prev[1] = seg[1]
        else:
            merged.append(seg)

    min_units_required = max(
        int(min_units_abs),
        int(np.ceil(min_units_frac * n_units)),
    )

    events = []

    for start_bin, end_bin in merged:
        start_s = start_bin * bin_s
        end_s = min((end_bin + 1) * bin_s, recording_duration_s)
        duration_s = end_s - start_s

        if duration_s < min_duration_s:
            continue

        burst_bins = set(range(start_bin, end_bin + 1))
        participating_units = []

        for unit_id, active_bins in zip(unit_ids, unit_active_bin_sets):
            if len(active_bins.intersection(burst_bins)) > 0:
                participating_units.append(unit_id)

        if len(participating_units) < min_units_required:
            continue

        peak_population_rate_hz = float(np.max(smoothed[start_bin:end_bin + 1]) / bin_s)

        events.append({
            "network_burst_id": len(events),
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "n_participating_units": len(participating_units),
            "participation_fraction": len(participating_units) / max(n_units, 1),
            "peak_population_rate_hz": peak_population_rate_hz,
            "participating_unit_ids": ";".join(map(str, participating_units)),
        })

    events_df = pd.DataFrame(events)

    population_rate_df = pd.DataFrame({
        "bin_start_s": bin_edges[:-1],
        "bin_end_s": bin_edges[1:],
        "population_spike_count": population_counts,
        "population_count_smoothed": smoothed,
        "network_burst_threshold": threshold,
        "above_threshold": above,
    })

    if len(events_df) == 0:
        empty_summary["network_threshold_population_count"] = threshold
        return empty_summary, events_df, population_rate_df

    summary = {
        "network_n_bursts": len(events_df),
        "network_burst_rate_hz": len(events_df) / max(recording_duration_s, 1e-12),
        "network_mean_burst_duration_s": float(events_df["duration_s"].mean()),
        "network_median_burst_duration_s": float(events_df["duration_s"].median()),
        "network_mean_participating_units": float(events_df["n_participating_units"].mean()),
        "network_mean_participation_fraction": float(events_df["participation_fraction"].mean()),
        "network_mean_peak_population_rate_hz": float(events_df["peak_population_rate_hz"].mean()),
        "network_threshold_population_count": threshold,
    }

    return summary, events_df, population_rate_df


def circular_shift_spikes(spikes, shift_s, t_start_s, t_stop_s):
    """
    Circularly shift spike times within [t_start_s, t_stop_s].
    """
    spikes = np.asarray(spikes, dtype=float)
    duration_s = t_stop_s - t_start_s

    if duration_s <= 0:
        return spikes

    shifted = ((spikes - t_start_s + shift_s) % duration_s) + t_start_s
    shifted.sort()

    return shifted


def compute_surrogate_tested_sttc(
    spike_times_list,
    unit_ids,
    recording_duration_s,
    dt_s=0.050,
    n_shuffles=50,
    alpha=0.05,
    min_spikes_per_unit=5,
    max_units=100,
    seed=0,
    progress_every_pairs=500,
):
    """
    Compute STTC and significant edges using circular-shift surrogates.

    Uses your existing sttc_correct() function.
    """
    rng = np.random.default_rng(seed)

    unit_ids = list(unit_ids)
    spike_times_list = [np.asarray(s, dtype=float) for s in spike_times_list]

    valid_indices = [
        idx for idx, spikes in enumerate(spike_times_list)
        if len(spikes) >= min_spikes_per_unit
    ]

    # Limit connectivity runtime using most active units.
    if len(valid_indices) > max_units:
        counts = np.array([len(spike_times_list[i]) for i in valid_indices])
        keep = np.argsort(counts)[::-1][:max_units]
        valid_indices = [valid_indices[i] for i in keep]

    included_unit_ids = [unit_ids[i] for i in valid_indices]
    included_spikes = [spike_times_list[i] for i in valid_indices]

    n = len(included_unit_ids)

    sttc_matrix = np.zeros((n, n), dtype=float)
    pvalue_matrix = np.ones((n, n), dtype=float)
    threshold_matrix = np.full((n, n), np.nan, dtype=float)
    significant_adjacency = np.zeros((n, n), dtype=bool)

    edge_rows = []

    if n < 2:
        return (
            sttc_matrix,
            pvalue_matrix,
            threshold_matrix,
            significant_adjacency,
            pd.DataFrame(),
            included_unit_ids,
        )

    total_pairs = n * (n - 1) // 2
    pair_counter = 0

    print(
        f"Computing surrogate-tested STTC for {n} units "
        f"({total_pairs} pairs, {n_shuffles} shuffles/pair, dt={dt_s}s)."
    )

    for i in range(n):
        for j in range(i + 1, n):
            pair_counter += 1

            if (
                progress_every_pairs is not None
                and progress_every_pairs > 0
                and pair_counter % int(progress_every_pairs) == 0
            ):
                print(
                    f"STTC progress: {pair_counter}/{total_pairs} pairs "
                    f"({100.0 * pair_counter / max(total_pairs, 1):.1f}%)"
                )

            ts1 = included_spikes[i]
            ts2 = included_spikes[j]

            real_sttc = sttc_correct(
                ts1,
                ts2,
                dt_s=dt_s,
                t_start_s=0.0,
                t_stop_s=recording_duration_s,
            )

            null_values = []

            for _ in range(n_shuffles):
                shift_s = rng.uniform(0.0, recording_duration_s)
                ts2_shifted = circular_shift_spikes(
                    ts2,
                    shift_s=shift_s,
                    t_start_s=0.0,
                    t_stop_s=recording_duration_s,
                )

                null_values.append(
                    sttc_correct(
                        ts1,
                        ts2_shifted,
                        dt_s=dt_s,
                        t_start_s=0.0,
                        t_stop_s=recording_duration_s,
                    )
                )

            null_values = np.asarray(null_values, dtype=float)
            threshold = float(np.nanpercentile(null_values, 100.0 * (1.0 - alpha)))
            p_value = float((1.0 + np.sum(null_values >= real_sttc)) / (n_shuffles + 1.0))
            significant = bool((real_sttc > threshold) and (p_value <= alpha))

            sttc_matrix[i, j] = real_sttc
            sttc_matrix[j, i] = real_sttc

            pvalue_matrix[i, j] = p_value
            pvalue_matrix[j, i] = p_value

            threshold_matrix[i, j] = threshold
            threshold_matrix[j, i] = threshold

            significant_adjacency[i, j] = significant
            significant_adjacency[j, i] = significant

            edge_rows.append({
                "unit_id_1": included_unit_ids[i],
                "unit_id_2": included_unit_ids[j],
                "sttc": real_sttc,
                "surrogate_threshold": threshold,
                "p_value": p_value,
                "significant_edge": significant,
                "dt_s": dt_s,
                "n_shuffles": n_shuffles,
            })

    edge_df = pd.DataFrame(edge_rows)

    return (
        sttc_matrix,
        pvalue_matrix,
        threshold_matrix,
        significant_adjacency,
        edge_df,
        included_unit_ids,
    )


def compute_graph_features(significant_adjacency, sttc_matrix, unit_ids):
    """
    Compute unit-level and well-level graph features from significant STTC edges.
    """
    unit_ids = list(unit_ids)
    adjacency = np.asarray(significant_adjacency, dtype=bool)
    sttc_matrix = np.asarray(sttc_matrix, dtype=float)

    n = len(unit_ids)

    empty_well = {
        "graph_n_nodes": n,
        "graph_n_edges": 0,
        "graph_density": 0.0,
        "graph_mean_degree": 0.0,
        "graph_average_clustering": np.nan,
        "graph_n_components": np.nan,
        "graph_largest_component_fraction": np.nan,
        "graph_modularity": np.nan,
        "significant_edge_fraction": 0.0,
    }

    if n == 0:
        return empty_well, pd.DataFrame()

    degree = adjacency.sum(axis=1)
    weighted_strength = np.nansum(sttc_matrix * adjacency, axis=1)

    unit_graph_df = pd.DataFrame({
        "graph_degree": degree,
        "graph_weighted_sttc_strength": weighted_strength,
    }, index=unit_ids)

    possible_edges = n * (n - 1) / 2.0
    n_edges = int(np.sum(adjacency) / 2)

    if nx is None:
        well_graph = {
            "graph_n_nodes": n,
            "graph_n_edges": n_edges,
            "graph_density": n_edges / max(possible_edges, 1.0),
            "graph_mean_degree": float(np.mean(degree)),
            "graph_average_clustering": np.nan,
            "graph_n_components": np.nan,
            "graph_largest_component_fraction": np.nan,
            "graph_modularity": np.nan,
            "significant_edge_fraction": n_edges / max(possible_edges, 1.0),
        }

        return well_graph, unit_graph_df

    G = nx.Graph()
    G.add_nodes_from(unit_ids)

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                G.add_edge(unit_ids[i], unit_ids[j], weight=float(sttc_matrix[i, j]))

    components = list(nx.connected_components(G))
    largest_component_fraction = (
        max(len(c) for c in components) / n if len(components) else 0.0
    )

    modularity = np.nan
    try:
        if G.number_of_edges() > 0:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            if len(communities) > 1:
                modularity = nx.algorithms.community.modularity(G, communities)
    except Exception:
        modularity = np.nan

    well_graph = {
        "graph_n_nodes": n,
        "graph_n_edges": G.number_of_edges(),
        "graph_density": float(nx.density(G)) if n > 1 else 0.0,
        "graph_mean_degree": float(np.mean(degree)),
        "graph_average_clustering": float(nx.average_clustering(G)) if n > 1 else 0.0,
        "graph_n_components": len(components),
        "graph_largest_component_fraction": float(largest_component_fraction),
        "graph_modularity": float(modularity) if np.isfinite(modularity) else np.nan,
        "significant_edge_fraction": G.number_of_edges() / max(possible_edges, 1.0),
    }

    return well_graph, unit_graph_df


def compute_windowed_unit_features(
    spike_times_list,
    unit_ids,
    recording_duration_s,
    windows,
):
    """
    Compute firing rate and ISI metrics in named perturbation windows.
    """
    rows = []

    for unit_id, spikes in zip(unit_ids, spike_times_list):
        spikes = np.asarray(spikes, dtype=float)

        for window in windows:
            window_name = window["window_name"]
            start_s = float(window.get("start_s", 0.0))
            end_s = window.get("end_s", None)

            if end_s is None:
                end_s = recording_duration_s
            else:
                end_s = float(end_s)

            if start_s >= recording_duration_s:
                continue

            end_s = min(end_s, recording_duration_s)

            if end_s <= start_s:
                continue

            window_spikes = spikes[(spikes >= start_s) & (spikes < end_s)]
            duration_s = max(end_s - start_s, 1e-12)

            features = compute_single_unit_firing_features(
                spike_times_s=window_spikes - start_s,
                recording_duration_s=duration_s,
            )

            features.update({
                "unit_id": unit_id,
                "window_name": window_name,
                "window_start_s": start_s,
                "window_end_s": end_s,
            })

            rows.append(features)

    return pd.DataFrame(rows)


def make_windowed_unit_features_wide(windowed_unit_df, baseline_window="baseline_0_5min"):
    """
    Convert long windowed features into wide unit-level response columns.

    Produces:
        fr_<window>
        delta_fr_<window>_vs_baseline
        pct_fr_<window>_vs_baseline
    """
    if windowed_unit_df is None or len(windowed_unit_df) == 0:
        return pd.DataFrame()

    rate_col = "firing_rate_from_spikes_hz"

    wide = windowed_unit_df.pivot_table(
        index="unit_id",
        columns="window_name",
        values=rate_col,
        aggfunc="mean",
    )

    wide.columns = [f"fr_{c}" for c in wide.columns]
    wide = wide.reset_index().set_index("unit_id")

    baseline_col = f"fr_{baseline_window}"

    if baseline_col in wide.columns:
        baseline = wide[baseline_col].replace(0, np.nan)

        for col in list(wide.columns):
            if col == baseline_col:
                continue

            clean_name = col.replace("fr_", "")
            wide[f"delta_fr_{clean_name}_vs_baseline"] = wide[col] - wide[baseline_col]
            wide[f"pct_fr_{clean_name}_vs_baseline"] = 100.0 * (
                wide[col] - wide[baseline_col]
            ) / baseline

    return wide

#%% ### OUTPUTS

def compute_quality_metrics_safe(analyzer, safe_id, output_dir = None):
    """
    Compute quality metrics with fallback for SpikeInterface version differences.
    """
    preferred_metrics = [
        "snr",
        "firing_rate",
        "presence_ratio",
        "isi_violation",
        "amplitude_cutoff",
    ]

    try:
        metrics = analyzer.compute(
            "quality_metrics",
            metric_names=preferred_metrics
        ).get_data()

    except Exception as e:
        print(f"[{safe_id}] WARNING: full metric set failed: {repr(e)}")
        print(f"[{safe_id}] Falling back to snr + firing_rate only.")

        metrics = analyzer.compute(
            "quality_metrics",
            metric_names=["snr", "firing_rate"]
        ).get_data()

    raw_metrics_path = os.path.join(
        output_dir,
        f"{safe_id}_quality_metrics_raw.csv"
    )

    if output_dir is not None:
        metrics.to_csv(raw_metrics_path, index_label="unit_id")

    if "snr" in metrics.columns:
        metrics["snr_invalid"] = ~np.isfinite(metrics["snr"])
        metrics["snr"] = metrics["snr"].replace([np.inf, -np.inf], np.nan)
    else:
        metrics["snr_invalid"] = True
        metrics["snr"] = np.nan

    return metrics


def make_well_summary(master_df):
    """
    Build well-level summary from a concatenated phenotype dataframe.
    Works for both per-raw-file and global master matrices.

    This version supports:
    - original firing/SNR/burst/STTC metrics
    - firing-pattern metrics
    - max-interval burst metrics
    - waveform metrics
    - spatial footprint metrics
    - graph/connectivity metrics
    - network-burst metrics

    It safely ignores columns that are not present yet.
    """

    summary_cols = {
        # Original core metrics
        "n_units": ("unit_id", "count"),
        "mean_firing_rate": ("firing_rate", "mean"),
        "median_firing_rate": ("firing_rate", "median"),
        "mean_snr": ("snr", "mean"),
        "median_snr": ("snr", "median"),
        "mean_burst_rate_hz": ("burst_rate_hz", "mean"),
        "median_burst_rate_hz": ("burst_rate_hz", "median"),
        "mean_network_sttc": ("mean_network_sttc", "mean"),
        "median_network_sttc": ("mean_network_sttc", "median"),

        # Firing / ISI
        "mean_n_spikes": ("n_spikes", "mean"),
        "median_n_spikes": ("n_spikes", "median"),
        "mean_firing_rate_from_spikes_hz": ("firing_rate_from_spikes_hz", "mean"),
        "median_firing_rate_from_spikes_hz": ("firing_rate_from_spikes_hz", "median"),
        "mean_mean_isi_s": ("mean_isi_s", "mean"),
        "mean_median_isi_s": ("median_isi_s", "mean"),
        "mean_std_isi_s": ("std_isi_s", "mean"),
        "mean_cv_isi": ("cv_isi", "mean"),
        "mean_lv_isi": ("lv_isi", "mean"),
        "mean_isi_entropy": ("isi_entropy", "mean"),
        "mean_refractory_violation_fraction_2ms": (
            "refractory_violation_fraction_2ms", "mean"
        ),
        "mean_silent_period_fraction_1s": (
            "silent_period_fraction_1s", "mean"
        ),

        # Max-interval bursting
        "mean_burst_rate_hz_max_interval": (
            "burst_rate_hz_max_interval", "mean"
        ),
        "median_burst_rate_hz_max_interval": (
            "burst_rate_hz_max_interval", "median"
        ),
        "mean_n_bursts_max_interval": (
            "n_bursts_max_interval", "mean"
        ),
        "median_n_bursts_max_interval": (
            "n_bursts_max_interval", "median"
        ),
        "mean_burst_duration_s_max_interval": (
            "mean_burst_duration_s_max_interval", "mean"
        ),
        "median_burst_duration_s_max_interval": (
            "median_burst_duration_s_max_interval", "median"
        ),
        "mean_spikes_per_burst_max_interval": (
            "mean_spikes_per_burst_max_interval", "mean"
        ),
        "median_spikes_per_burst_max_interval": (
            "median_spikes_per_burst_max_interval", "median"
        ),
        "mean_fraction_spikes_in_bursts_max_interval": (
            "fraction_spikes_in_bursts_max_interval", "mean"
        ),
        "mean_interburst_interval_s_max_interval": (
            "mean_interburst_interval_s_max_interval", "mean"
        ),
        "mean_burst_duration_cv_max_interval": (
            "burst_duration_cv_max_interval", "mean"
        ),

        # Waveform
        "mean_template_p2p_max_uV": (
            "template_p2p_max_uV", "mean"
        ),
        "median_template_p2p_max_uV": (
            "template_p2p_max_uV", "median"
        ),
        "mean_template_trough_to_peak_ms": (
            "template_trough_to_peak_ms", "mean"
        ),
        "median_template_trough_to_peak_ms": (
            "template_trough_to_peak_ms", "median"
        ),
        "mean_template_half_width_ms": (
            "template_half_width_ms", "mean"
        ),
        "median_template_half_width_ms": (
            "template_half_width_ms", "median"
        ),
        "mean_template_repolarization_slope_uV_per_ms": (
            "template_repolarization_slope_uV_per_ms", "mean"
        ),

        # Spatial footprint
        "mean_template_footprint_n_channels": (
            "template_footprint_n_channels", "mean"
        ),
        "median_template_footprint_n_channels": (
            "template_footprint_n_channels", "median"
        ),
        "mean_template_footprint_radius_um": (
            "template_footprint_radius_um", "mean"
        ),
        "median_template_footprint_radius_um": (
            "template_footprint_radius_um", "median"
        ),
        "mean_template_footprint_area_um2": (
            "template_footprint_area_um2", "mean"
        ),
        "median_template_footprint_area_um2": (
            "template_footprint_area_um2", "median"
        ),

        # Connectivity / graph unit-level summaries
        "mean_graph_degree": (
            "graph_degree", "mean"
        ),
        "median_graph_degree": (
            "graph_degree", "median"
        ),
        "mean_graph_weighted_sttc_strength": (
            "graph_weighted_sttc_strength", "mean"
        ),
        "median_graph_weighted_sttc_strength": (
            "graph_weighted_sttc_strength", "median"
        ),

        # Network-burst well-level summaries
        # These are copied onto each unit row, so use first.
        "network_n_bursts": (
            "network_n_bursts", "first"
        ),
        "network_burst_rate_hz": (
            "network_burst_rate_hz", "first"
        ),
        "network_mean_burst_duration_s": (
            "network_mean_burst_duration_s", "first"
        ),
        "network_median_burst_duration_s": (
            "network_median_burst_duration_s", "first"
        ),
        "network_mean_participating_units": (
            "network_mean_participating_units", "first"
        ),
        "network_mean_participation_fraction": (
            "network_mean_participation_fraction", "first"
        ),
        "network_mean_peak_population_rate_hz": (
            "network_mean_peak_population_rate_hz", "first"
        ),

        # Graph well-level summaries
        # These are copied onto each unit row, so use first.
        "graph_n_nodes": (
            "graph_n_nodes", "first"
        ),
        "graph_n_edges": (
            "graph_n_edges", "first"
        ),
        "graph_density": (
            "graph_density", "first"
        ),
        "graph_mean_degree": (
            "graph_mean_degree", "first"
        ),
        "graph_average_clustering": (
            "graph_average_clustering", "first"
        ),
        "graph_n_components": (
            "graph_n_components", "first"
        ),
        "graph_largest_component_fraction": (
            "graph_largest_component_fraction", "first"
        ),
        "graph_modularity": (
            "graph_modularity", "first"
        ),
        "significant_edge_fraction": (
            "significant_edge_fraction", "first"
        ),
    }

    # Optional original quality metrics
    if "presence_ratio" in master_df.columns:
        summary_cols["mean_presence_ratio"] = ("presence_ratio", "mean")

    if "isi_violation" in master_df.columns:
        summary_cols["mean_isi_violation"] = ("isi_violation", "mean")

    if "isi_violations_ratio" in master_df.columns:
        summary_cols["mean_isi_violations_ratio"] = (
            "isi_violations_ratio", "mean"
        )

    if "amplitude_cutoff" in master_df.columns:
        summary_cols["mean_amplitude_cutoff"] = (
            "amplitude_cutoff", "mean"
        )

    # Very important:
    # Drop requested summary columns that are not present yet.
    # This lets the same function work for old and new phenotype matrices.
    summary_cols = {
        out_col: spec
        for out_col, spec in summary_cols.items()
        if spec[0] in master_df.columns
    }

    group_cols = [
        "raw_file_id",
        "raw_relative_path",
        "recording_id",
        "well_id",
        "condition",
        "celltype",
    ]

    # Optional metadata columns if you add metadata merging later.
    optional_group_cols = [
        "cell_line",
        "donor_id",
        "genotype",
        "disease",
        "treatment",
        "dose",
        "timepoint",
        "DIV",
        "plate_id",
        "batch_id",
        "operator",
        "recording_temperature",
        "baseline_or_post_treatment",
    ]

    group_cols = group_cols + optional_group_cols

    group_cols = [
        col for col in group_cols
        if col in master_df.columns
    ]

    well_summary = master_df.groupby(
        group_cols,
        dropna=False
    ).agg(**summary_cols).reset_index()

    return well_summary
