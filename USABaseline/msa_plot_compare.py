import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================
# Argument Parsing
# =============================
def parse_args():
    parser = argparse.ArgumentParser(description="Compare MSA and USA projections and fits.")
    parser.add_argument('--usa_baseline', type=str, required=True, help='Path to USA baseline output CSV')
    parser.add_argument('--msa_baseline', type=str, required=True, help='Path to MSA baseline output CSV')
    parser.add_argument('--msa_calibration', type=str, required=True, help='Path to MSA calibration output CSV')
    parser.add_argument('--msa_name', type=str, required=True, help='MSA name (cs_name) to plot')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional: directory to save plots')
    return parser.parse_args()

# =============================
# Data Loading and Filtering
# =============================
def load_and_filter_data(usa_path, msa_baseline_path, msa_calib_path, msa_name, end_date):
    # Load data
    usa = pd.read_csv(usa_path, parse_dates=['Date'], infer_datetime_format=True)
    msa_base = pd.read_csv(msa_baseline_path, parse_dates=['Year_Month_Day'], infer_datetime_format=True)
    msa_calib = pd.read_csv(msa_calib_path, parse_dates=['Year_Month_Day'], infer_datetime_format=True)

    # Filter by end_date
    end_dt = pd.to_datetime(end_date)
    usa = usa[usa['Date'] <= end_dt].copy()
    msa_base = msa_base[msa_base['Year_Month_Day'] <= end_dt].copy()
    msa_calib = msa_calib[msa_calib['Year_Month_Day'] <= end_dt].copy()

    # Filter for the specified MSA
    msa_base_msa = msa_base[msa_base['cs_name'] == msa_name].copy()
    msa_calib_msa = msa_calib[msa_calib['cs_name'] == msa_name].copy()

    if msa_base_msa.empty or msa_calib_msa.empty:
        raise ValueError(f"No data found for MSA '{msa_name}' in one or more files.")

    return usa, msa_base_msa, msa_calib_msa

def plot_msa_usa_comparison(usa, msa_base, msa_calib, msa_name, end_date, output_dir=None):
    # Prepare data
    # MSA actuals: use hpa12m or HPI (actual), baseline: ProjectedHPA1YFwd_MSABaseline, calibration: ProjectedHPA1YFwd_MSA
    # USA actuals/projections: try to use USA_HPA1Yfwd, ProjectedHPA1YFwd_USABaseline
    date_col = 'Year_Month_Day'
    msa_base = msa_base.sort_values(date_col)
    msa_calib = msa_calib.sort_values(date_col)
    
    # MSA actuals (use hpa12m if available, else HPI)
    if 'hpa12m' in msa_base.columns:
        msa_actual = msa_base['hpa12m']
        actual_label = 'MSA Actual (hpa12m)'
    elif 'HPI' in msa_base.columns:
        msa_actual = msa_base['HPI']
        actual_label = 'MSA Actual (HPI)'
    else:
        msa_actual = None
        actual_label = 'MSA Actual'
    dates = msa_base[date_col]
    # Baseline projections
    msa_baseline_proj = msa_base['ProjectedHPA1YFwd_MSABaseline'] if 'ProjectedHPA1YFwd_MSABaseline' in msa_base.columns else None
    # Calibration projections
    msa_calib_proj = msa_calib['ProjectedHPA1YFwd_MSA'] if 'ProjectedHPA1YFwd_MSA' in msa_calib.columns else None
    # USA projections and actuals
    usa_proj = None
    usa_actual = None
    usa_dates = None
    if 'ProjectedHPA1YFwd_USABaseline' in msa_base.columns:
        usa_proj = msa_base['ProjectedHPA1YFwd_USABaseline']
        usa_dates = msa_base[date_col]
    elif 'ProjectedHPA1YFwd_USABaseline' in msa_calib.columns:
        usa_proj = msa_calib['ProjectedHPA1YFwd_USABaseline']
        usa_dates = msa_calib[date_col]
    if 'USA_HPA1Yfwd' in msa_base.columns:
        usa_actual = msa_base['USA_HPA1Yfwd']
    elif 'USA_HPA1Yfwd' in msa_calib.columns:
        usa_actual = msa_calib['USA_HPA1Yfwd']
    # Plot
    plt.figure(figsize=(14,7))
    if msa_actual is not None:
        plt.plot(dates, msa_actual, label=actual_label, color='black', linewidth=2)
    if msa_baseline_proj is not None:
        plt.plot(dates, msa_baseline_proj, label='MSA Baseline Projection', color='blue', linestyle='--')
    if msa_calib_proj is not None:
        plt.plot(msa_calib[date_col], msa_calib_proj, label='MSA Recalibrated Projection', color='red', linestyle='-')
    if usa_proj is not None and usa_dates is not None:
        plt.plot(usa_dates, usa_proj, label='USA Baseline Projection', color='green', linestyle='--')
    if usa_actual is not None and usa_dates is not None:
        plt.plot(usa_dates, usa_actual, label='USA Actual', color='orange', linestyle='-')
    # Mark last 12 months as projections
    if len(dates) >= 12:
        last12 = dates.iloc[-12:]
        plt.axvspan(last12.iloc[0], last12.iloc[-1], color='gray', alpha=0.15, label='Projection Period (Last 12M)')
    plt.title(f"MSA vs USA Fit and Projections for {msa_name}")
    plt.xlabel('Date')
    plt.ylabel('HPA / Projection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"msa_usa_projection_{msa_name.replace(' ','_')}_{end_date}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=200)
    plt.show()

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

def print_msa_metrics(msa_base, msa_calib, msa_name):
    # Only use train period (tag == 'Train')
    if 'tag' not in msa_base.columns or 'tag' not in msa_calib.columns:
        print("[WARN] 'tag' column not found in one of the files. Skipping fit metrics.")
        return
    base_train = msa_base[msa_base['tag'] == 'Train']
    calib_train = msa_calib[msa_calib['tag'] == 'Train']
    # Actuals
    if 'hpa12m' in base_train.columns:
        y_true = base_train['hpa12m']
    elif 'HPI' in base_train.columns:
        y_true = base_train['HPI']
    else:
        print("[WARN] No actuals found for MSA fit metrics.")
        return
    # Baseline fit
    if 'ProjectedHPA1YFwd_MSABaseline' in base_train.columns:
        y_pred_base = base_train['ProjectedHPA1YFwd_MSABaseline']
    else:
        y_pred_base = None
    # Calibration fit
    if 'ProjectedHPA1YFwd_MSA' in calib_train.columns:
        y_pred_calib = calib_train['ProjectedHPA1YFwd_MSA']
    else:
        y_pred_calib = None
    print(f"\nModel Fit Metrics for {msa_name} (Train period):")
    if y_pred_base is not None:
        metrics_base = compute_metrics(y_true, y_pred_base)
        print("  Baseline:", metrics_base)
    if y_pred_calib is not None:
        metrics_calib = compute_metrics(y_true, y_pred_calib)
        print("  Recalibrated:", metrics_calib)

def print_overall_metrics(msa_base_all, msa_calib_all):
    # Only use train period (tag == 'Train')
    if 'tag' not in msa_base_all.columns or 'tag' not in msa_calib_all.columns:
        print("[WARN] 'tag' column not found in one of the files. Skipping overall metrics.")
        return
    base_train = msa_base_all[msa_base_all['tag'] == 'Train']
    calib_train = msa_calib_all[msa_calib_all['tag'] == 'Train']
    # Actuals
    if 'hpa12m' in base_train.columns:
        y_true = base_train['hpa12m']
    elif 'HPI' in base_train.columns:
        y_true = base_train['HPI']
    else:
        print("[WARN] No actuals found for overall fit metrics.")
        return
    # Baseline fit
    if 'ProjectedHPA1YFwd_MSABaseline' in base_train.columns:
        y_pred_base = base_train['ProjectedHPA1YFwd_MSABaseline']
    else:
        y_pred_base = None
    # Calibration fit
    if 'ProjectedHPA1YFwd_MSA' in calib_train.columns:
        y_pred_calib = calib_train['ProjectedHPA1YFwd_MSA']
    else:
        y_pred_calib = None
    print(f"\nOverall Model Fit Metrics (All MSAs, Train period):")
    if y_pred_base is not None:
        metrics_base = compute_metrics(y_true, y_pred_base)
        print("  Baseline:", metrics_base)
    if y_pred_calib is not None:
        metrics_calib = compute_metrics(y_true, y_pred_calib)
        print("  Recalibrated:", metrics_calib)

def compute_per_msa_metrics(msa_base_all, msa_calib_all, output_dir=None, end_date=None):
    # Get all unique MSAs
    msas = sorted(set(msa_base_all['cs_name'].unique()) & set(msa_calib_all['cs_name'].unique()))
    rows = []
    for msa in msas:
        base = msa_base_all[(msa_base_all['cs_name'] == msa) & (msa_base_all['tag'] == 'Train')]
        calib = msa_calib_all[(msa_calib_all['cs_name'] == msa) & (msa_calib_all['tag'] == 'Train')]
        if 'hpa12m' in base.columns:
            y_true = base['hpa12m']
        elif 'HPI' in base.columns:
            y_true = base['HPI']
        else:
            continue
        if 'ProjectedHPA1YFwd_MSABaseline' in base.columns:
            y_pred_base = base['ProjectedHPA1YFwd_MSABaseline']
        else:
            y_pred_base = None
        if 'ProjectedHPA1YFwd_MSA' in calib.columns:
            y_pred_calib = calib['ProjectedHPA1YFwd_MSA']
        else:
            y_pred_calib = None
        row = {'MSA': msa}
        if y_pred_base is not None:
            m_base = compute_metrics(y_true, y_pred_base)
            row.update({f'Baseline_{k}': v for k, v in m_base.items()})
        if y_pred_calib is not None:
            m_calib = compute_metrics(y_true, y_pred_calib)
            row.update({f'Recalibrated_{k}': v for k, v in m_calib.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if output_dir and end_date:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"msa_fit_metrics_{end_date}.csv"
        df.to_csv(os.path.join(output_dir, fname), index=False)
        print(f"[INFO] Per-MSA metrics exported to {os.path.join(output_dir, fname)}")
    return df

def plot_rmse_bar(metrics_df, output_dir=None, end_date=None):
    # Only plot if both columns exist
    if 'Baseline_RMSE' not in metrics_df.columns or 'Recalibrated_RMSE' not in metrics_df.columns:
        print("[WARN] RMSE columns not found in metrics table. Skipping bar plot.")
        return
    df = metrics_df.sort_values('Baseline_RMSE', ascending=False)
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(max(10, len(df)//4), 6))
    plt.bar(x - width/2, df['Baseline_RMSE'], width, label='Baseline RMSE', color='blue', alpha=0.7)
    plt.bar(x + width/2, df['Recalibrated_RMSE'], width, label='Recalibrated RMSE', color='red', alpha=0.7)
    plt.xticks(x, df['MSA'], rotation=90)
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison for All MSAs (Baseline vs Recalibrated)')
    plt.legend()
    plt.tight_layout()
    if output_dir and end_date:
        fname = f"msa_rmse_bar_{end_date}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=200)
        print(f"[INFO] RMSE bar plot saved to {os.path.join(output_dir, fname)}")
    plt.show()

# =============================
# Main Entrypoint (rest to be added in next chunks)
# =============================
def main():
    args = parse_args()
    usa, msa_base, msa_calib = load_and_filter_data(
        args.usa_baseline, args.msa_baseline, args.msa_calibration, args.msa_name, args.end_date
    )
    plot_msa_usa_comparison(usa, msa_base, msa_calib, args.msa_name, args.end_date, args.output_dir)
    print_msa_metrics(msa_base, msa_calib, args.msa_name)
    msa_base_all = pd.read_csv(args.msa_baseline, parse_dates=['Year_Month_Day'], infer_datetime_format=True)
    msa_calib_all = pd.read_csv(args.msa_calibration, parse_dates=['Year_Month_Day'], infer_datetime_format=True)
    print_overall_metrics(msa_base_all, msa_calib_all)
    metrics_df = compute_per_msa_metrics(msa_base_all, msa_calib_all, args.output_dir, args.end_date)
    print("\nSample of per-MSA fit metrics:")
    print(metrics_df.head())
    plot_rmse_bar(metrics_df, args.output_dir, args.end_date)

if __name__ == "__main__":
    main() 