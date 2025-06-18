import pandas as pd
import numpy as np
from pathlib import Path

def create_matrix():
    print("Loading the dataset dataframe...")
    # Make sure this path points to your main dataframe file
    df_path = Path("cholec80_processed/dataframes/cholec_split_250px_25fps.pkl")
    if not df_path.exists():
        print(f"Error: Dataframe file not found at {df_path}")
        print("Please run utils/tecno/create_dataframe.py first.")
        return

    df = pd.read_pickle(df_path)

    # Define the order of phases and tools to match your model
    phases = [
        "Preparation", "CalotTriangleDissection", "ClippingCutting", 
        "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", 
        "GallbladderRetraction"
    ]
    tools = [
        "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", 
        "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
    ]

    # Initialize a 7x7 matrix with zeros
    incompatibility_matrix = np.zeros((len(tools), len(phases)), dtype=np.int32)

    print("Analyzing tool and phase relationships...")
    for phase_idx, phase_name in enumerate(phases):
        # Get all frames belonging to the current phase
        phase_df = df[df['class'] == phase_idx]

        for tool_idx, tool_name in enumerate(tools):
            # If the sum of a tool's usage during a phase is 0, it was never used.
            if phase_df[tool_name].sum() == 0:
                print(f"-> Found incompatibility: Tool '{tool_name}' is NEVER used in phase '{phase_name}'")
                incompatibility_matrix[tool_idx, phase_idx] = 1 # Mark as incompatible

    output_path = "tool_phase_incompatibility.npy"
    np.save(output_path, incompatibility_matrix)
    print(f"\nIncompatibility matrix created and saved to '{output_path}'")
    print("Matrix (Tools x Phases):\n", incompatibility_matrix)

if __name__ == "__main__":
    create_matrix()