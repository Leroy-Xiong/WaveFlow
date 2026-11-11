import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns

from modules.utils import WAVE_VARIABLES, ERA5_VARIABLE_NAMES, MAJOR_CITIES
try:
    from modules.plot.satellite_background import add_satellite_background, add_satellite_background_global
except ImportError:
    add_satellite_background = None
    add_satellite_background_global = None
def plot_rmse_comparison(rmse_csv_files, labels=None, save_path="rmse_comparison.png", title="RMSE Comparison"):
    """
    Plot boxplot comparison of RMSE from multiple sample sets.
    
    Args:
        rmse_csv_files: List of paths to RMSE CSV files, or dict with {label: path}
        labels: List of labels for each sample set (optional if rmse_csv_files is dict)
        save_path: Path to save the comparison plot
        title: Title of the plot
    """
    
    # Handle dict input format
    if isinstance(rmse_csv_files, dict):
        labels = list(rmse_csv_files.keys())
        csv_files = list(rmse_csv_files.values())
    else:
        csv_files = rmse_csv_files
        # Generate default labels if not provided
        if labels is None:
            labels = [f"Sample {i+1}" for i in range(len(csv_files))]
    
    # Validate input
    if len(csv_files) != len(labels):
        print(f"Error: Number of files ({len(csv_files)}) doesn't match number of labels ({len(labels)})")
        return
    
    if len(csv_files) == 0:
        print("Error: No CSV files provided")
        return
    
    # Load and combine all data
    df_list = []
    stats = []
    
    for csv_file, label in zip(csv_files, labels):
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: File not found: {csv_file}")
            return
        
        # Load data
        try:
            df = pd.read_csv(csv_file)
            df['Group'] = label
            df_list.append(df)
            
            # Calculate statistics
            stats.append({
                'label': label,
                'mean': df['rmse'].mean(),
                'median': df['rmse'].median(),
                'std': df['rmse'].std()
            })
        except Exception as e:
            print(f"Error loading CSV file {csv_file}: {e}")
            return
    
    # Combine all dataframes
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Create plot with dynamic figure size
    fig_width = max(10, len(csv_files) * 2)
    plt.figure(figsize=(fig_width, 6))
    
    # Boxplot
    sns.boxplot(x='Group', y='rmse', data=df_combined, width=0.5, palette="Set3", showfliers=False, hue='Group', legend=False)
    
    # Create title with all means
    mean_text = " | ".join([f"{s['label']}: {s['mean']:.4f}" for s in stats])
    # plt.title(f"{title}\nMean RMSE: {mean_text}")
    plt.title(f"{title}")
    plt.ylabel("RMSE")
    plt.xlabel("Method")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Save plot
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ RMSE comparison plot saved to {save_path}")
    for s in stats:
        print(f"  {s['label']}: Mean={s['mean']:.4f}, Median={s['median']:.4f}, Std={s['std']:.4f}")


def plot_multi_variable_rmse_comparison(rmse_csv_files, labels=None, save_path="multi_variable_rmse_comparison.png", title="RMSE Comparison by Variable", variables=None):
    """
    Polished boxplot comparing RMSE distributions across groups for the
    specified wave variables. Layout auto-adapts: 4→2×2, 3→1×3, 2→1×2, 1→1×1.

    Visual features:
      - Seaborn `whitegrid` theme, perceptually-uniform `viridis` palette
        (color order matches input order so it conveys magnitude/ordinal info).
      - Bold median line, white diamond mean marker.
      - Translucent strip/scatter overlay of individual samples for distribution
        intuition (auto-thinned for large N).
      - Unit-aware y-axis label per variable (from WAVE_VARIABLES).
      - Y-limit clipped at the 99th percentile across groups to suppress
        long-tail outliers without losing the box body.
      - Compact "Mean RMSE" stats table anchored inside each subplot.
      - Adaptive figure size and x-label rotation based on group count.

    Args:
        rmse_csv_files: List of paths to CSV files, or dict {label: path}.
        labels: Optional list of labels (ignored if rmse_csv_files is a dict).
        save_path: Where to save the figure.
        title: Top-level figure title.
        variables: Variables to plot. Default: ['hs', 'tp', 'dm', 'elev'].
    """
    from matplotlib.offsetbox import AnchoredText
    import numpy as np

    # ---------------- Input handling ----------------
    if isinstance(rmse_csv_files, dict):
        labels = list(rmse_csv_files.keys())
        csv_files = list(rmse_csv_files.values())
    else:
        csv_files = rmse_csv_files
        if labels is None:
            labels = [f"Sample {i+1}" for i in range(len(csv_files))]

    if len(csv_files) != len(labels):
        print(f"Error: Number of files ({len(csv_files)}) doesn't match number of labels ({len(labels)})")
        return
    if len(csv_files) == 0:
        print("Error: No CSV files provided")
        return

    if variables is None:
        variables = ['hs', 'tp', 'dm', 'elev']
    target_vars = list(variables)
    combined_data = {var: [] for var in target_vars}
    stats_summary = {var: [] for var in target_vars}

    # ---------------- Load CSVs ----------------
    for csv_file, label in zip(csv_files, labels):
        if not os.path.exists(csv_file):
            print(f"Error: File not found: {csv_file}")
            continue
        try:
            df = pd.read_csv(csv_file)
            for var in target_vars:
                col_name = var
                if col_name not in df.columns:
                    if f"rmse_{var}" in df.columns:
                        col_name = f"rmse_{var}"
                    elif f"{var}_rmse" in df.columns:
                        col_name = f"{var}_rmse"
                    else:
                        continue
                values = df[col_name].dropna().values
                sub_df = pd.DataFrame({'rmse': values, 'Group': label})
                combined_data[var].append(sub_df)
                stats_summary[var].append({
                    'label': label,
                    'mean': float(df[col_name].mean()),
                    'median': float(df[col_name].median()),
                    'std': float(df[col_name].std()),
                })
        except Exception as e:
            print(f"Error loading CSV file {csv_file}: {e}")
            return

    # ---------------- Style ----------------
    n_groups = len(labels)
    n_vars = len(target_vars)
    palette = sns.color_palette("viridis", n_colors=max(n_groups, 2))

    # Adaptive layout: 4→1×4, 3→1×3, 2→1×2, 1→1×1
    if n_vars <= 4:
        n_rows, n_cols = 1, n_vars
    else:
        n_cols = min(n_vars, 4)
        n_rows = int(np.ceil(n_vars / n_cols))

    # Figure size scales with group count and variable count
    fig_w = max(n_cols * 7, 2.6 * n_groups + 4)
    fig_h = max(n_rows * 14, 14)

    with sns.axes_style("whitegrid"), plt.rc_context({
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#444444',
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
    }):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        if n_vars == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, var in enumerate(target_vars):
            ax = axes[i]
            if not combined_data[var]:
                ax.text(0.5, 0.5, f"No data for {var}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(var)
                continue

            df_var = pd.concat(combined_data[var], ignore_index=True)

            # Per-variable labels: only include groups that have data for this variable
            var_groups = [g for g in labels if g in df_var["Group"].unique()]
            var_palette = [palette[labels.index(g)] for g in var_groups]

            # Boxplot: bold median + clean caps
            sns.boxplot(
                x='Group', y='rmse', data=df_var,
                order=var_groups,
                width=0.55,
                hue='Group',
                hue_order=var_groups,
                palette=var_palette,
                showfliers=False,
                legend=False,
                linewidth=1.4,
                medianprops={'color': 'black', 'linewidth': 2.4},
                whiskerprops={'color': '#444', 'linewidth': 1.2},
                capprops={'color': '#444', 'linewidth': 1.2},
                boxprops={'edgecolor': '#222', 'linewidth': 1.2},
                ax=ax,
            )

            # Strip overlay (thinned automatically) for sample-level cloud
            n_per_group = max((df_var.groupby('Group').size()).max(), 1)
            sample_frac = 1.0 if n_per_group <= 600 else 600.0 / n_per_group
            if sample_frac < 1.0:
                # pandas 3.0: groupby().apply() drops grouping columns; use concat instead
                _chunks = []
                for _g, _d in df_var.groupby('Group'):
                    _chunks.append(_d.sample(frac=sample_frac, random_state=0))
                df_strip = pd.concat(_chunks, ignore_index=True)
            else:
                df_strip = df_var
            sns.stripplot(
                x='Group', y='rmse', data=df_strip,
                order=var_groups,
                size=2.2, jitter=0.18, alpha=0.18,
                color='#222222',
                ax=ax,
            )

            # Mean diamond markers
            means = [s['mean'] for s in stats_summary[var]]
            ax.scatter(
                range(len(means)), means,
                marker='D', s=55, edgecolors='black', linewidths=1.0,
                facecolors='white', zorder=5, label='Mean',
            )

            # Y limit: clip at 99th percentile across all data of this var
            y_hi = float(np.percentile(df_var['rmse'].values, 99))
            if np.isnan(y_hi) or np.isinf(y_hi):
                y_hi = 0.0
            y_lo = max(0.0, float(df_var['rmse'].min()) - 0.02 * y_hi)
            if np.isnan(y_lo) or np.isinf(y_lo):
                y_lo = 0.0
            ax.set_ylim(y_lo, y_hi * 1.05)

            # Labels
            var_info = WAVE_VARIABLES.get(var, {})
            plot_label = var_info.get('plot_label', var)
            unit = var_info.get('units', '')
            ax.set_title(plot_label, pad=10)
            ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
            ax.set_xlabel("")

            # Adaptive x tick rotation (per-variable group count)
            n_var_groups = len(var_groups)
            rot = 0 if n_var_groups < 3 else 25
            ax.tick_params(axis='x', rotation=rot)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment('right' if rot else 'center')

            # No per-subplot annotation

        if title:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98] if not title else [0, 0.02, 1, 0.96])

        # Save
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Multi-variable RMSE comparison plot saved to {save_path}")

    # Print detailed stats
    for var in target_vars:
        if stats_summary[var]:
            print(f"\nStats for {var}:")
            for s in stats_summary[var]:
                print(f"  {s['label']}: Mean={s['mean']:.4f}, Median={s['median']:.4f}, Std={s['std']:.4f}")


