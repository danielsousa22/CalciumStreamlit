import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks

st.set_page_config(page_title="Calcium Transient Analyzer", layout="wide")
# Constants
FRAME_RATE = 500  # frames per second

# Helper: compute features for one file
@st.experimental_memo
def compute_features(df, smooth, peaks, start_s, end_s):
    times = df['frame'] / FRAME_RATE
    # intervals & BPM
    if len(peaks) > 1:
        intervals = np.diff(peaks / FRAME_RATE)
        avg_bpm = 60 / intervals.mean()
        mean_pp = intervals.mean()
        std_pp = intervals.std()
    else:
        avg_bpm = mean_pp = std_pp = np.nan
    # choose a central peak for window default if not batching
    # analysis window
    start_idx = np.searchsorted(times, start_s)
    end_idx = np.searchsorted(times, end_s)
    seg_times = times.iloc[start_idx:end_idx].values
    seg_signal = smooth[start_idx:end_idx]
    # baseline & peak
    peak_idx = peaks[0] if len(peaks)>0 else start_idx
    peak_time = df['frame'].iloc[peak_idx] / FRAME_RATE
    baseline = np.percentile(seg_signal[seg_times<=peak_time], 10)
    peak_val = smooth[peak_idx]
    amplitude = peak_val - baseline
    time_to_peak = peak_time - start_s
    # decay metrics
    decay_stats = {}
    for pct in [0.9, 0.5, 0.1]:
        thresh = baseline + pct * amplitude
        post = seg_signal[seg_times>=peak_time]
        post_t = seg_times[seg_times>=peak_time]
        idxs = np.where(post <= thresh)[0]
        decay_stats[f'decay_to_{int(pct*100)}%'] = (post_t[idxs[0]] - peak_time) if idxs.size else np.nan
    # pack features
    data = {
        'baseline': baseline,
        'amplitude': amplitude,
        'time_to_peak': time_to_peak,
        'average_bpm': avg_bpm,
        'mean_peak_to_peak_s': mean_pp,
        'std_peak_to_peak_s': std_pp,
        **decay_stats
    }
    return pd.DataFrame([data])

# Sidebar navigation
st.sidebar.title("Navigation")
step = st.sidebar.radio("Go to step:", [
    "1. Upload & Smooth",
    "2. Peak Detection",
    "3. Feature Calculation",
    "4. Batch Analysis"
])

# Session state init
for key in ['raw_df','smoothed','peaks','avg_bpm','mean_pp_interval','std_pp_interval','analysis_window']:
    if key not in st.session_state:
        st.session_state[key] = None
# Step 1: Upload & Smooth
if step == "1. Upload & Smooth":
    st.header("Step 1: Upload and Smooth Signal")
    uploaded = st.file_uploader("Upload CSV file (frame, intensity)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        df.columns = ['frame', 'intensity']
        st.session_state.raw_df = df.copy()
        st.subheader("Raw Signal")
        df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        time = df['frame'] / FRAME_RATE
        # Plot raw signal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=df['intensity'], mode='lines', name='Raw'))
        fig.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence Intensity', height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Smoothing controls
        st.subheader("Smoothing Parameters")
        window = st.number_input("Window length (odd)", min_value=3, max_value=101, value=25, step=2)
        poly = st.number_input("Polynomial order", min_value=1, max_value=5, value=3, step=1)
        if st.button("Update Smoothed Signal"):
            smooth = savgol_filter(df['intensity'], window_length=window, polyorder=poly)
            st.session_state.smoothed = smooth

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=time, y=df['intensity'], mode='lines', name='Raw', opacity=0.3))
            fig2.add_trace(go.Scatter(x=time, y=smooth, mode='lines', name='Smoothed'))
            fig2.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence Intensity', height=600)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Please upload a CSV file to continue.")

# Step 2: Peak Detection
elif step == "2. Peak Detection":
    st.header("Step 2: Peak Detection")
    df = st.session_state.raw_df
    smooth = st.session_state.smoothed
    if df is None or smooth is None:
        st.warning("Step 1 must be completed first.")
    else:
        st.subheader("Smoothed Signal")
        df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
        time = df['frame'] / FRAME_RATE

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=time, y=smooth, mode='lines', name='Smoothed'))
        fig3.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence Intensity', height=600)
        st.plotly_chart(fig3, use_container_width=True)

        # Peak detection controls
        st.subheader("Detection Parameters")
        height = st.number_input("Min Peak Height", value=100.0)
        distance = st.number_input("Min Distance (frames)", value=300)
        skip_first = st.checkbox("Skip first peak")
        skip_last = st.checkbox("Skip last peak")
        if st.button("Update Detection"):
            peaks, props = find_peaks(smooth, height=height, distance=distance)
            sel = peaks.copy()
            if skip_first and len(sel)>0:
                sel = sel[1:]
            if skip_last and len(sel)>1:
                sel = sel[:-1]
            st.session_state.peaks = sel
            # Calculate average BPM
            times = sel / FRAME_RATE
            if len(times) > 1:
                intervals = np.diff(times)
                st.session_state.avg_bpm = 60 / intervals.mean()
                st.session_state.mean_pp_interval = intervals.mean()
                st.session_state.std_pp_interval = intervals.std()
            else:
                st.session_state.avg_bpm = 0
                st.session_state.mean_pp_interval = 0
                st.session_state.std_pp_interval = 0

            # Plot with peaks
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=time, y=smooth, mode='lines', name='Smoothed'))
            peak_times = df['frame'].iloc[sel] / FRAME_RATE
            fig4.add_trace(go.Scatter(x=peak_times, y=smooth[sel], mode='markers', marker=dict(color='red', size=6), name='Peaks'))

            fig4.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence Intensity', height=600)
            st.plotly_chart(fig4, use_container_width=True)

            st.write(f"Detected {len(sel)} peaks; Average BPM: {st.session_state.avg_bpm:.2f}")
        st.info("Adjust parameters and click 'Update Detection'.")

# --- Step 3: Feature Calculation ---
else:
    st.header("Step 3: Feature Calculation")
    df = st.session_state.raw_df
    smooth = st.session_state.smoothed
    peaks = st.session_state.peaks
    avg_bpm = st.session_state.avg_bpm
    mean_pp = st.session_state.mean_pp_interval
    std_pp = st.session_state.std_pp_interval
    if df is None or smooth is None or len(peaks)==0:
        st.warning("Complete Steps 1 & 2, and detect at least one peak, before calculating features.")
    else:
        # Peak selection
        peak_labels = [f"Frame {int(df['frame'].iloc[p])} ({df['frame'].iloc[p]/FRAME_RATE:.3f}s)" for p in peaks]
        sel_idx = st.selectbox("Select peak to analyze", options=list(range(len(peaks))), format_func=lambda i: peak_labels[i])
        peak_frame = peaks[sel_idx]
        peak_time = df['frame'].iloc[peak_frame]/FRAME_RATE

        # Prepare analysis window defaults
        default_start = max(0.0, peak_time - 0.5)
        default_end = peak_time + 0.8
        times_full = df['frame']/FRAME_RATE
        min_time = float(times_full.min())
        max_time = float(times_full.max())
        if st.session_state.analysis_window is None:
            st.session_state.analysis_window = (default_start, default_end)

        # Window selection form
        with st.form(key="window_form"):
            start_time, end_time = st.slider(
                "Select analysis window (s)", min_value=min_time, max_value=max_time,
                value=tuple(st.session_state.analysis_window), step=(max_time-min_time)/1000)
            recalc = st.form_submit_button("Recalculate Features")
            if recalc:
                st.session_state.analysis_window = (start_time, end_time)

        # Use session_state window for calculations
        start_time, end_time = st.session_state.analysis_window
        st.write(f"Current window: {start_time:.3f}s to {end_time:.3f}s")

        # Convert to indices
        start_idx = np.searchsorted(times_full, start_time)
        end_idx = np.searchsorted(times_full, end_time)

        seg_times = times_full.iloc[start_idx:end_idx].values
        seg_signal = smooth[start_idx:end_idx]

        # Feature calculations
        baseline = np.percentile(seg_signal[seg_times<=peak_time], 10)
        peak_val = smooth[peak_frame]
        amplitude = peak_val - baseline
        time_to_peak = peak_time - start_time

        decay_stats = {}
        for pct in [0.9, 0.5, 0.1]:
            thresh_level = baseline + pct*amplitude
            post = seg_signal[seg_times>=peak_time]
            post_times = seg_times[seg_times>=peak_time]
            idxs = np.where(post<=thresh_level)[0]
            decay_stats[f'decay_to_{int(pct*100)}%'] = (post_times[idxs[0]]-peak_time) if len(idxs)>0 else np.nan

        # Prepare result table
        result_df = pd.DataFrame({
            "baseline": [baseline],
            "amplitude": [amplitude],
            "time_to_peak": [time_to_peak],
            **{k: [v] for k, v in decay_stats.items()},
            "average_bpm": [avg_bpm],
            "mean_peak_to_peak_s": [mean_pp],
            "std_peak_to_peak_s": [std_pp]
        })
        st.subheader("Calculated Features")
        st.table(result_df)

        # Plot segmented peak with markers
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=seg_times, y=seg_signal, mode='lines', name='Segment'))
        fig5.add_trace(go.Scatter(x=[start_time], y=[baseline], mode='markers', marker=dict(color='green', size=8), name='Baseline'))
        fig5.add_trace(go.Scatter(x=[peak_time], y=[peak_val], mode='markers', marker=dict(color='red', size=10), name='Peak'))
        for pct, dt in decay_stats.items():
            if not np.isnan(dt):
                t = peak_time + dt
                level = baseline + (int(pct.split('_')[2][:-1])/100)*amplitude
                fig5.add_trace(go.Scatter(x=[t], y=[level], mode='markers', marker=dict(symbol='x', size=8), name=pct))
        fig5.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence Intensity', height=600)
        st.plotly_chart(fig5, use_container_width=True)

        # Download options
        csv_summary = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download features CSV", csv_summary, "calcium_features.csv", "text/csv")
        segment_df = pd.DataFrame({'time_s': seg_times, 'intensity': seg_signal})
        csv_segment = segment_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download segment data CSV", csv_segment, "segmented_peak_data.csv", "text/csv")
        img_bytes = fig5.to_image(format="png")
        st.download_button("Download segment plot PNG", img_bytes, "segmented_peak_plot.png", "image/png")
# Step 4: Batch Analysis
if step == "4. Batch Analysis":
    st.header("Step 4: Batch Analysis")
    files = st.file_uploader(
        "Upload CSV files for batch", type='csv', accept_multiple_files=True
    )
    manifest = st.file_uploader(
        "Upload manifest CSV (filename,height,distance,start_s,end_s)",
        type='csv', key='manifest'
    )
    if st.button("Run batch"):
        if not files or not manifest:
            st.error("Please upload both data files and a manifest.")
        else:
            params = pd.read_csv(manifest)
            results = []
            for f in files:
                df = pd.read_csv(f, header=None, names=['frame','intensity'])
                df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
                # smoothing: use same default or add to manifest if desired
                smooth = savgol_filter(df['intensity'], window_length=25, polyorder=3)
                # find parameters for this file
                row = params[params['filename']==f.name]
                if row.empty:
                    st.warning(f"No params for {f.name}, skipping.")
                    continue
                h = float(row['height'])
                d = int(row['distance'])
                start_s = float(row['start_s'])
                end_s = float(row['end_s'])
                peaks, _ = find_peaks(smooth, height=h, distance=d)
                feat_df = compute_features(df, smooth, peaks, start_s, end_s)
                feat_df['filename'] = f.name
                results.append(feat_df)
            if results:
                batch_df = pd.concat(results, ignore_index=True)
                st.subheader("Batch Results")
                st.dataframe(batch_df)
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download batch results CSV", csv, "batch_results.csv", "text/csv"
                )
            else:
                st.info("No results to show.")
