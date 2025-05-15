import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

def parse_report(file_path):
    """Parse the sulci report file into a dataframe, handling (X, Y, Z): format."""
    # Initialize lists to store data
    sulci = []
    occurrence_rates = []
    point_counts = []
    mean_x = []
    mean_y = []
    mean_z = []
    sd_x = []
    sd_y = []
    sd_z = []
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for sulcus line
        if line.startswith("Sulcus ") and ":" in line:
            # Extract sulcus number
            current_sulcus = int(line.split()[1].replace(":", ""))
            sulci.append(current_sulcus)
            
            # Extract occurrence rate
            i += 1
            if i < len(lines) and "Occurrence Rate:" in lines[i]:
                rate = float(lines[i].split(":")[1].replace("%", "").strip())
                occurrence_rates.append(rate)
            else:
                sulci.pop()  # Remove the sulcus if we can't find its occurrence rate
                i -= 1  # Go back if not found
                continue
            
            # Extract points count
            i += 1
            if i < len(lines) and "Points Count:" in lines[i]:
                count = int(lines[i].split(":")[1].strip())
                point_counts.append(count)
            else:
                sulci.pop()  # Remove the sulcus
                occurrence_rates.pop()
                i -= 1  # Go back if not found
                continue
            
            # Extract mean position - handle the (X, Y, Z): (values) format
            i += 1
            if i < len(lines) and "Mean Position" in lines[i]:
                try:
                    # Extract the part after the second colon and inside parentheses
                    values_part = lines[i].split(":", 1)[1].strip()
                    if values_part.startswith("(") and values_part.endswith(")"):
                        values = values_part[1:-1].split(",")
                        if len(values) == 3:
                            mean_x.append(float(values[0].strip()))
                            mean_y.append(float(values[1].strip()))
                            mean_z.append(float(values[2].strip()))
                        else:
                            raise ValueError("Expected 3 values for mean position")
                    else:
                        raise ValueError("Mean position values not in parentheses")
                except (IndexError, ValueError) as e:
                    print(f"Error parsing mean position for sulcus {current_sulcus}: {e}")
                    sulci.pop()
                    occurrence_rates.pop()
                    point_counts.pop()
                    i -= 1
                    continue
            else:
                sulci.pop()
                occurrence_rates.pop() 
                point_counts.pop()
                i -= 1  # Go back if not found
                continue
            
            # Extract std dev position - handle the (X, Y, Z): (values) format
            i += 1
            if i < len(lines) and "Std. Dev. Position" in lines[i]:
                try:
                    # Extract the part after the second colon and inside parentheses
                    values_part = lines[i].split(":", 1)[1].strip()
                    if values_part.startswith("(") and values_part.endswith(")"):
                        values = values_part[1:-1].split(",")
                        if len(values) == 3:
                            sd_x.append(float(values[0].strip()))
                            sd_y.append(float(values[1].strip()))
                            sd_z.append(float(values[2].strip()))
                        else:
                            raise ValueError("Expected 3 values for std dev position")
                    else:
                        raise ValueError("Std dev position values not in parentheses")
                except (IndexError, ValueError) as e:
                    print(f"Error parsing std dev position for sulcus {current_sulcus}: {e}")
                    sulci.pop()
                    occurrence_rates.pop()
                    point_counts.pop()
                    mean_x.pop()
                    mean_y.pop()
                    mean_z.pop()
                    i -= 1
                    continue
            else:
                sulci.pop()
                occurrence_rates.pop()
                point_counts.pop()
                mean_x.pop()
                mean_y.pop()
                mean_z.pop()
                i -= 1  # Go back if not found
                continue
        
        i += 1
    
    # Calculate variability score
    variability = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(sd_x, sd_y, sd_z)]
    
    # Create dataframe
    data = {
        'Sulcus': sulci,
        'Occurrence_Rate': occurrence_rates,
        'Points_Count': point_counts,
        'Mean_X': mean_x,
        'Mean_Y': mean_y,
        'Mean_Z': mean_z,
        'SD_X': sd_x,
        'SD_Y': sd_y,
        'SD_Z': sd_z,
        'Variability_Score': variability
    }
    
    df = pd.DataFrame(data)
    df['Hemisphere'] = df['Sulcus'].apply(lambda x: 'Left' if x <= 30 else 'Right')
    # Sulcus name map
    sulcus_name_map = {
        0: 'o', 50: 'o',
        1: 'W', 51: 'W',
        2: 'r1', 52: 'r1',
        3: 'fo', 53: 'fo',
        4: 'fs', 54: 'fs',
        5: 'fm', 55: 'fm',
        6: 'h', 56: 'h',
        7: 'fi', 57: 'fi',
        8: 'pc', 58: 'pc',
        9: 'c', 59: 'c',
        10: 'pt', 60: 'pt',
        11: 'ip', 61: 'ip',
        12: 'otr', 62: 'otr',
        13: 'oci', 63: 'oci',
        14: 'L', 64: 'L',
        15: 'PreL', 65: 'PreL',
        16: 'lc', 66: 'lc',
        17: 'rc', 67: 'rc',
        18: 'po', 68: 'po',
        19: 'ts', 69: 'ts',
        20: 'ti', 70: 'ti',
        21: 'S', 71: 'S',
        22: 'hr', 72: 'hr',
        23: 'ar', 73: 'ar',
        24: 'd', 74: 'd',
        25: 'r2', 75: 'r2',
        26: 'col', 76: 'col',
        27: 'ot', 77: 'ot',
        28: 'loc', 78: 'loc',
        30: 'other', 80: 'other'
    }
    
    # Add a 'Sulcus_Label' column like "o (L)"
    df['Sulcus_Label'] = df.apply(
        lambda row: f"{sulcus_name_map.get(row['Sulcus'], '?')} ({row['Hemisphere'][0]})", axis=1
    )

    return df

def plot_variability(df, output_dir):
    """Create simple plots of sulci variability."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by variability
    df_sorted = df.sort_values('Sulcus')
    colors = df_sorted['Hemisphere'].map({'Left': 'cornflowerblue', 'Right': 'firebrick'})
    
    # Plot variability ranking
    plt.figure(figsize=(14, 8))
    colors = df_sorted['Hemisphere'].map({'Left': 'cornflowerblue', 'Right': 'firebrick'})  # define color mapping
    plt.bar(df_sorted['Sulcus_Label'], df_sorted['Variability_Score'], color=colors)
    
    plt.title('Sulci Ranked by Positional Variability (Colour by Hemisphere)')
    plt.xlabel('Sulcus ID')
    plt.ylabel('Variability Score')
    plt.xticks(rotation=90)
    
    # Add this legend block here
    legend_elements = [
        Patch(facecolor='cornflowerblue', label='Left Hemisphere'),
        Patch(facecolor='firebrick', label='Right Hemisphere')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Final layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variability_by_hemisphere_coloured.png'), dpi=300)
    plt.close()
    
    # Plot X, Y, Z variability comparison for top 15 most variable sulci
    top15 = df.nlargest(15, 'Variability_Score')
    
    plt.figure(figsize=(12, 8))
    ind = np.arange(len(top15))
    width = 0.25
    
    plt.bar(ind - width, top15['SD_X'], width, label='X')
    plt.bar(ind, top15['SD_Y'], width, label='Y')
    plt.bar(ind + width, top15['SD_Z'], width, label='Z')
    
    plt.xlabel('Sulcus ID')
    plt.ylabel('Standard Deviation')
    plt.title('X, Y, Z Variability of 15 Most Variable Sulci')
    xtick_labels = top15['Sulcus_Label']
    xtick_colors = top15['Hemisphere'].map({'Left': 'cornflowerblue', 'Right': 'firebrick'}).values
    
    ax = plt.gca()
    ax.set_xticks(ind)
    ax.set_xticklabels(xtick_labels, rotation=90)

    # Colour each x tick label
    for ticklabel, color in zip(ax.get_xticklabels(), xtick_colors):
        ticklabel.set_color(color)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xyz_comparison.png'), dpi=300)
    plt.close()
    
    # Histogram of Composite Variability
    plt.figure(figsize=(10, 6))
    plt.hist(df['Variability_Score'], bins=20, color='slateblue', edgecolor='black')
    plt.axvline(df['Variability_Score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(df['Variability_Score'].quantile(0.9), color='orange', linestyle='--', linewidth=2, label='90th Percentile')
    
    plt.title('Distribution of Composite Variability Scores')
    plt.xlabel('Composite Variability')
    plt.ylabel('Number of Sulci')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variability_distribution_histogram.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

# Generate a Word-friendly table
def generate_word_friendly_table(df, output_file):
    """Generate a table formatted for copying into Microsoft Word."""
    
    # Select and rename columns for the table
    table_df = df[['Sulcus', 'Hemisphere', 'Occurrence_Rate', 'Points_Count', 'SD_X', 'SD_Y', 'SD_Z', 
                   'Variability_Score']].copy()
    
    # Rename columns for presentation
    table_df.columns = ['Sulcus ID', 'Hemisphere', 'Occurrence Rate (%)', 'Point Count',
                        'SD X', 'SD Y', 'SD Z', 
                        'Composite Variability']
    
    # Format numeric columns
    table_df['Occurrence Rate (%)'] = table_df['Occurrence Rate (%)'].round(1)
    table_df['SD X'] = table_df['SD X'].round(1)
    table_df['SD Y'] = table_df['SD Y'].round(1)
    table_df['SD Z'] = table_df['SD Z'].round(1)
    table_df['Composite Variability'] = table_df['Composite Variability'].round(1)
    
    # Sort by variability (most consistent first)
    table_df = table_df.sort_values('Composite Variability')
    
    # Rank the sulci
    table_df['Rank'] = range(1, len(table_df) + 1)
    # Reorder columns to put rank first
    table_df = table_df[['Rank', 'Sulcus ID', 'Occurrence Rate (%)', 'Point Count',
                         'SD X', 'SD Y', 'SD Z', 'Composite Variability']]
    
    # Save as CSV (can be imported into Word)
    table_df.to_csv(output_file, index=False)
    
    # Print a formatted version that can be copied directly
    print("\nSulcal Imprint Variability (sorted from most to least consistent)")
    print("=" * 110)
    print(table_df.to_string(index=False))
    print("=" * 110)
    print(f"Table saved to {output_file} - you can also copy the above formatted table")
    
    return table_df

def generate_paired_variability_table(df):
    # Mapping of sulcus IDs to labels
    sulcus_pairs = {
        'o': (0, 50), 'W': (1, 51), 'r': (2, 52), 'fo': (3, 53),
        'fs': (4, 54), 'fm': (5, 55), 'h': (6, 56), 'fi': (7, 57),
        'pc': (8, 58), 'c': (9, 59), 'pt': (10, 60), 'ip': (11, 61),
        'otr': (12, 62), 'oci': (13, 63), 'L': (14, 64), 'PreL': (15, 65),
        'lc': (16, 66), 'rc': (17, 67), 'po': (18, 68), 'ts': (19, 69),
        'ti': (20, 70), 'S': (21, 71), 'hr': (22, 72), 'ar': (23, 73),
        'd': (24, 74), 'rh': (25, 75), 'col': (26, 76), 'ot': (27, 77),
        'loc': (28, 78), 'other': (30, 80)
    }

    rows = []
    for label, (left_id, right_id) in sulcus_pairs.items():
        left = df[df['Sulcus'] == left_id]
        right = df[df['Sulcus'] == right_id]

        left_var = left['Variability_Score'].values[0] if not left.empty else None
        right_var = right['Variability_Score'].values[0] if not right.empty else None

        if left_var is not None and right_var is not None:
            diff = abs(left_var - right_var)
            mean = (left_var + right_var) / 2
        else:
            diff = None
            mean = None

        rows.append({
            'Sulcus': label,
            'Left Variability': left_var,
            'Right Variability': right_var,
            'Difference': diff,
            'Mean Variability': mean
        })

    paired_df = pd.DataFrame(rows)
    return paired_df

# Main function
def main():
    # File path
    report_file = "F:/Samples/All human/Endocast density map pipeline parallel processing/Results/sulci_report.txt"
    
    # Try with a file dialog if needed
    if not os.path.exists(report_file):
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        report_file = askopenfilename(title="Select Sulci Report File", 
                                     filetypes=[("Text Files", "*.txt")])
        if not report_file:
            print("No file selected. Exiting...")
            return
    
    print(f"Reading from {report_file}...")
    
    # Parse the file
    df = parse_report(report_file)
    
    if df.empty:
        print("No data extracted. Please check the file format.")
        return
    
    print(f"Extracted data for {len(df)} sulci")
    
    # Create table
    output_file = "F:/Samples/All human/Endocast density map pipeline parallel processing/Results/sulci_variability_table.csv"
    table_df = generate_word_friendly_table(df, output_file)
    
    # Create plots
    output_dir = "F:/Samples/All human/Endocast density map pipeline parallel processing/Results/variability_plots"
    plot_variability(df, output_dir)
    
    # Create paired sulcus table
    paired_df = generate_paired_variability_table(df)
    paired_csv = os.path.join(output_dir, 'sulcus_paired_variability.csv')
    paired_df.to_csv(paired_csv, index=False)
    print("\nPaired sulcus variability saved to:", paired_csv)
    print(paired_df)
    
    print("\nAnalysis complete!")
    
    # Find the top and bottom ranked sulci
    most_consistent_id = table_df.iloc[0]['Sulcus ID']
    most_variable_id = table_df.iloc[-1]['Sulcus ID']
    most_consistent_var = table_df.iloc[0]['Composite Variability']
    most_variable_var = table_df.iloc[-1]['Composite Variability']
    
    # Look up labels from the full df
    label_lookup = df.set_index('Sulcus')['Sulcus_Label'].to_dict()
    label_consistent = label_lookup.get(most_consistent_id, str(most_consistent_id))
    label_variable = label_lookup.get(most_variable_id, str(most_variable_id))
    
    print("\nSummary:")
    print(f"- Most consistent sulcus: {label_consistent} (Variability: {most_consistent_var})")
    print(f"- Most variable sulcus: {label_variable} (Variability: {most_variable_var})")



if __name__ == "__main__":
    main()