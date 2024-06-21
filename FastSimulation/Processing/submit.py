import os
import subprocess

# Directory containing the files
# Kaons
files_directory = "/Cherenkov/ParticleGun/gen_k/monitoring_hists/json"
# Pions
files_directory = "/Cherenkov/ParticleGun/gen_pi/monitoring_hists/json"

# Path to the Python script to be executed
python_script = "/sciclone/home/jgiroux/Cherenkov_FastSim/Processing/make_GlueX_data.py"

# Directory to store submission scripts
submission_scripts_directory = "/sciclone/home/jgiroux/Cherenkov_FastSim/Processing/scripts"

# Create the submission scripts directory if it doesn't exist
os.makedirs(submission_scripts_directory, exist_ok=True)

os.makedirs(os.path.join(files_directory,'processed'),exist_ok=True)

# Iterate over the files in the directory
i = 0
for filename in os.listdir(files_directory):
    # Check if the item is a file
    if os.path.isfile(os.path.join(files_directory, filename)):
        # Construct the full file path
        file_path = os.path.join(files_directory, filename)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to megabytes
        ram_per_job = int(file_size_mb * 2)  # Set RAM to be 2 times the size of the file
        # If needed you can do dynamic memory allocation in jobs. Just used fixed memory for now to prevent I/O issues.
        
        # Create a submission script for each file
        submission_script = f"""\
#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name={os.path.splitext(filename)[0]}
#SBATCH --nodes=1
#SBATCH --tasks=6
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu=2000

module load anaconda3/2021.11
source activate ptorch

python {python_script} --file {file_path} --base_dir {files_directory}
"""

        # Write the submission script to a file in the submission scripts directory
        submission_script_path = os.path.join(submission_scripts_directory, f"{os.path.splitext(filename)[0]}.sh")
        with open(submission_script_path, 'w') as f:
            f.write(submission_script)
            
        # Execute the submission script
        subprocess.run(["sbatch", submission_script_path])
    # i += 1
    # if i == 3:
    #     break  # For testing purposes, remove this line to process all files

