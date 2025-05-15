import os
import json
import glob

# Define directories
data_dir = "C:/Users/edwin/OneDrive - University of Cambridge/Cambridge/PhD/Python code/Endocast density map pipeline/Data"
results_dir = "C:/Users/edwin/OneDrive - University of Cambridge/Cambridge/PhD/Python code/Endocast density map pipeline/Results"
reference_dir = os.path.join(data_dir, "Reference")
visualisations_dir = "C:/Users/edwin/OneDrive - University of Cambridge/Cambridge/PhD/Python code/Endocast density map pipeline/Visualisations"

# Define file patterns
brain_scans_pattern = os.path.join(data_dir, "*.nii")
sulci_curves_pattern = os.path.join(data_dir, "*_curves_lab.csv")  # Adjust this pattern to match your sulci files
ref_surface_file = os.path.join(reference_dir, "Pp_F_M3_13201_OK.vtk")  # Adjust to your reference surface filename

# Find all files matching the patterns
brain_scan_files = sorted(glob.glob(brain_scans_pattern))
sulci_curve_files = sorted(glob.glob(sulci_curves_pattern))

# Display what was found
print(f"Found {len(brain_scan_files)} brain scan files")
print(f"Found {len(sulci_curve_files)} sulci curve files")

# Make sure we have matching brain scans and sulci curves
if len(brain_scan_files) != len(sulci_curve_files):
    print("Warning: Number of brain scans and sulci curves doesn't match!")
    print("Please ensure you have corresponding scan and curve files.")

# Check if reference file exists
if os.path.exists(ref_surface_file):
    print(f"Found reference surface file: {ref_surface_file}")
else:
    print(f"Warning: Reference surface file not found at {ref_surface_file}")
    # Try to find any VTK file in the reference directory
    vtk_files = glob.glob(os.path.join(reference_dir, "*.vtk"))
    if vtk_files:
        ref_surface_file = vtk_files[0]
        print(f"Using found VTK file instead: {ref_surface_file}")
    else:
        print("No VTK files found in the Reference directory.")

# Load existing config
with open('my_config.json', 'r') as f:
    config = json.load(f)

# Update the config
config['input_image_files'] = brain_scan_files
config['input_curve_files'] = sulci_curve_files
if os.path.exists(ref_surface_file):
    config['ref_surface_file'] = ref_surface_file

# Update output directory
config['output_dir'] = results_dir

# Add visualisations settings
if 'debug' not in config:
    config['debug'] = {}
    
config['debug']['visualization'] = {
    'enable': True,
    'save_images': True,
    'display_slices': False,
    'output_dir': visualisations_dir  # Add this to specify visualizations output directory
}

# Save the updated config
with open('my_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated configuration file with detected files")
print("Please review my_config.json to ensure it's correct before running the analysis")