import os
import numpy as np
import SimpleITK as sitk
import glob
from brain_shape_analysis.visualization import BrainVisualization
from brain_shape_analysis.sulci_analysis_tools import SulciAnalysisTools

# Define paths to your result files
results_dir = "C:/Users/edwin/OneDrive - University of Cambridge/Cambridge/PhD/Python code/Endocast density map pipeline/Results"
vis_output_dir = "C:/Users/edwin/OneDrive - University of Cambridge/Cambridge/PhD/Python code/Endocast density map pipeline/Visualisations/new"

# Make sure the output directory exists
os.makedirs(vis_output_dir, exist_ok=True)

# Load results
average_im = sitk.GetArrayFromImage(sitk.ReadImage(f"{results_dir}/Result_defImage.nii"))
var_x = sitk.GetArrayFromImage(sitk.ReadImage(f"{results_dir}/Result_Vx.nii"))
var_y = sitk.GetArrayFromImage(sitk.ReadImage(f"{results_dir}/Result_Vy.nii"))
var_z = sitk.GetArrayFromImage(sitk.ReadImage(f"{results_dir}/Result_Vz.nii"))

# Load probability maps from Result_DefPoints_Label_*.nii files
prob_maps = {}
prob_map_files = glob.glob(f"{results_dir}/Result_DefPoints_Label_*.nii")
for file_path in prob_map_files:
    # Extract label number from filename
    label = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
    # Load probability map
    prob_map = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    prob_maps[label] = prob_map
    print(f"Loaded probability map for label {label}")

# Create visualization object
viz = BrainVisualization(vis_output_dir)

# Generate new visualizations
print("Generating average shape visualization...")
viz.visualize_average_shape(average_im, var_x, var_y, var_z)

# Generate sulci probability maps visualizations
if prob_maps:
    print(f"Generating {len(prob_maps)} sulci probability map visualizations...")
    viz.visualize_sulci_probability_maps(average_im, prob_maps)
    
    # Generate 3D rendering
    print("Generating 3D rendering...")
    viz.create_3d_rendering(average_im, prob_maps)
else:
    print("No probability maps found.")

print("New visualizations generated!")