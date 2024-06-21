#!/bin/bash

# Create the json folder if it doesn't exist
mkdir -p /Cherenkov/ParticleGun/json

# Iterate over all .root files in the current directory
for root_file in /Cherenkov/ParticleGun/*.root; do
    # Extract the filename without extension
    filename=$(basename "$root_file")
    filename_noext="${filename%.*}"

    # Construct the output JSON filename in the json folder
    json_file="/Cherenkov/ParticleGun/json/${filename_noext}.json"

    # Execute the root command with the appropriate parameters
    root -q DrcHit.cc+ DrcEvent.cc+ 'MakeDictionaries.C("'"${root_file}"'", "'"${json_file}"'")'
    
done

