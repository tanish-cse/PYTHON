# Weather Data Visualizer - Mini Project
# Course: Programming for Problem Solving using Python
# Student: Ayush
#
# This script:
# 1) Loads weather data from a CSV file using pandas
# 2) Cleans and processes the data (dates, NaN values, etc.)
# 3) Computes statistics using numpy and pandas groupby
# 4) Creates multiple plots using matplotlib and saves them as PNGs
# 5) Exports cleaned data to a new CSV file
# 6) Generates a text report with basic insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File names (you can change if needed)
# -----------------------------
RAW_FILE = "weather_data.csv"
CLEANED_FILE = "weather_data_cleaned.csv"
REPORT_FILE = "weather_summary_report.txt"


# -----------------------------
# Task 1: Data Acquisition and Loading
# -----------------------------
def load_data(file_path):
    """
    Load CSV file into a pandas DataFrame and print basic info.
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    print("\n--- HEAD (first 5 rows) ---")
    print(df.head())

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- DESCRIBE ---")
    print(df.describe(include="all"))

    return df


# -----------------------------
# Task 2: Data Cleaning and Processing
# -----------------------------
def clean_data(df):
    """
    Clean the weather data:
    - Convert Date to datetime
    - Handle missing values
    - Keep only relevant columns
    - Add Year and Month columns
    """
    df = df.copy()

    # Convert Date column to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("Expected a 'Date' column in the CSV.")

    # Drop rows where date conversion failed
    df = df.dropna(subset=["Date"])

    # Keep only relevant columns (if they exist)
    columns_to_keep = ["Date"]
    if "Temperature" in df.columns:
        columns_to_keep.append("Temperature")
    if "Rainfall" in df.columns:
        columns_to_keep.append("Rainfall")
    if "Humidity" in df.columns:
        columns_to_keep.append("Humidity")

    df = df[columns_to_keep]

    # Handle missing numeric values by filling with column mean
    for col in df.columns:
        if df[col].dtype != "datetime64[ns]":
            df[col] = df[col].astype(float)
            df[col] = df[col].fillna(df[col].mean())

    # Add year and month columns for grouping
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    print("\n--- CLEANED DATA (first 5 rows) ---")
    print(df.head())

    return df


# -----------------------------
# Task 3: Statistical Analysis with NumPy
# -----------------------------
def compute_statistics(df):
    """
    Compute basic statistics using numpy and pandas.
    Daily stats are basically the existing rows.
    We will also compute monthly and yearly stats.
    """
    stats = {}

    if "Temperature" in df.columns:
        temp = df["Temperature"].values
        stats["temp_mean"] = np.mean(temp)
        stats["temp_min"] = np.min(temp)
        stats["temp_max"] = np.max(temp)
        stats["temp_std"] = np.std(temp)
    else:
        print("Warning: 'Temperature' column not found.")
        return stats, None, None

    # Monthly statistics
    monthly_stats = (
        df.groupby(["Year", "Month"])["Temperature"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )

    # Yearly statistics
    yearly_stats = (
        df.groupby("Year")["Temperature"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )

    print("\n--- MONTHLY TEMPERATURE STATS ---")
    print(monthly_stats.head())

    print("\n--- YEARLY TEMPERATURE STATS ---")
    print(yearly_stats.head())

    return stats, monthly_stats, yearly_stats


# -----------------------------
# Task 4: Visualization with Matplotlib
# -----------------------------
def create_plots(df):
    """
    Create and save:
    - Line chart: daily temperature trend
    - Bar chart: monthly rainfall totals
    - Scatter: humidity vs temperature
    - Combined figure with two subplots
    """
    # Make sure Date is sorted
    df = df.sort_values("Date")

    # 1) Line chart - daily temperature
    if "Temperature" in df.columns:
        plt.figure()
        plt.plot(df["Date"], df["Temperature"])
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.title("Daily Temperature Trend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("daily_temperature_trend.png")
        plt.close()
        print("Saved: daily_temperature_trend.png")

    # 2) Bar chart - monthly rainfall totals
    if "Rainfall" in df.columns:
        df_monthly_rain = (
            df.set_index("Date")
            .resample("M")["Rainfall"]
            .sum()
            .reset_index()
        )

        plt.figure()
        plt.bar(df_monthly_rain["Date"].dt.to_period("M").astype(str),
                df_monthly_rain["Rainfall"])
        plt.xlabel("Month")
        plt.ylabel("Total Rainfall (mm)")
        plt.title("Monthly Rainfall Totals")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("monthly_rainfall_totals.png")
        plt.close()
        print("Saved: monthly_rainfall_totals.png")

    # 3) Scatter plot - humidity vs temperature
    if "Humidity" in df.columns and "Temperature" in df.columns:
        plt.figure()
        plt.scatter(df["Temperature"], df["Humidity"])
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Humidity (%)")
        plt.title("Humidity vs Temperature")
        plt.tight_layout()
        plt.savefig("humidity_vs_temperature.png")
        plt.close()
        print("Saved: humidity_vs_temperature.png")

    # 4) Combined figure with two plots (subplots)
    if "Temperature" in df.columns and "Rainfall" in df.columns:
        df_monthly = df.set_index("Date").resample("M").agg(
            {"Temperature": "mean", "Rainfall": "sum"}
        ).reset_index()

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # Subplot 1: line chart for temperature
        axes[0].plot(df_monthly["Date"], df_monthly["Temperature"])
        axes[0].set_title("Average Monthly Temperature")
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].tick_params(axis='x', rotation=45)

        # Subplot 2: bar chart for rainfall
        axes[1].bar(df_monthly["Date"].dt.to_period("M").astype(str),
                    df_monthly["Rainfall"])
        axes[1].set_title("Monthly Rainfall")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Rainfall (mm)")
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig("combined_temperature_rainfall.png")
        plt.close()
        print("Saved: combined_temperature_rainfall.png")


# -----------------------------
# Task 5: Grouping and Aggregation
# -----------------------------
def add_season_column(df):
    """
    Simple season mapping based on month.
    (You can change according to your location if needed.)
    """
    def month_to_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Summer"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-Monsoon"

    df = df.copy()
    df["Season"] = df["Month"].apply(month_to_season)
    return df


def group_and_aggregate(df):
    """
    Group data by month and season and calculate aggregate statistics.
    """
    monthly_group = df.groupby(["Year", "Month"]).agg(
        {
            "Temperature": "mean",
            "Rainfall": "sum",
            "Humidity": "mean"
        }
    ).reset_index()

    df_season = add_season_column(df)
    seasonal_group = df_season.groupby("Season").agg(
        {
            "Temperature": "mean",
            "Rainfall": "sum",
            "Humidity": "mean"
        }
    ).reset_index()

    print("\n--- MONTHLY GROUPED DATA (first 5 rows) ---")
    print(monthly_group.head())

    print("\n--- SEASONAL GROUPED DATA ---")
    print(seasonal_group)

    return monthly_group, seasonal_group


# -----------------------------
# Task 6: Export and Storytelling
# -----------------------------
def export_cleaned_data(df, filename=CLEANED_FILE):
    """
    Save cleaned data to a new CSV file.
    """
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to: {filename}")


def write_report(stats, monthly_group, seasonal_group, filename=REPORT_FILE):
    """
    Write a summary report as a text file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Weather Data Analysis Report\n")
        f.write("===========================\n\n")

        if stats:
            f.write("Overall Temperature Statistics:\n")
            f.write(f"- Mean: {stats['temp_mean']:.2f} °C\n")
            f.write(f"- Min: {stats['temp_min']:.2f} °C\n")
            f.write(f"- Max: {stats['temp_max']:.2f} °C\n")
            f.write(f"- Std Dev: {stats['temp_std']:.2f}\n\n")

        f.write("Monthly Average (first few rows):\n")
        f.write(str(monthly_group.head()))
        f.write("\n\n")

        f.write("Seasonal Summary:\n")
        f.write(str(seasonal_group))
        f.write("\n\n")

        f.write("Basic Interpretation:\n")
        f.write("- Check which months have highest rainfall.\n")
        f.write("- Observe which season has highest average temperature.\n")
        f.write("- Check if higher temperatures relate to lower or higher humidity.\n")

    print(f"Report saved to: {filename}")


# -----------------------------
# MAIN FUNCTION: Run the full pipeline
# -----------------------------
def main():
    # Check if file exists
    if not Path(RAW_FILE).exists():
        print(f"Error: {RAW_FILE} not found in current folder.")
        print("Please place your weather_data.csv file in the same directory as this script.")
        return

    # Step 1: Load
    df_raw = load_data(RAW_FILE)

    # Step 2: Clean
    df_clean = clean_data(df_raw)

    # Step 3: Statistics
    stats, monthly_stats, yearly_stats = compute_statistics(df_clean)

    # Step 4: Grouping
    monthly_group, seasonal_group = group_and_aggregate(df_clean)

    # Step 5: Plots
    create_plots(df_clean)

    # Step 6: Export cleaned data
    export_cleaned_data(df_clean, CLEANED_FILE)

    # Step 7: Report
    write_report(stats, monthly_group, seasonal_group, REPORT_FILE)

    print("\nAll tasks completed. Check generated PNGs, cleaned CSV, and report file.")


if __name__ == "__main__":
    main()