#!/bin/tcsh

# Create the json folder if it doesn't exist
mkdir -p /Cherenkov/ParticleGun/json

# Iterate over all .root files in the current directory
foreach root_file ( /Cherenkov/ParticleGun/*.root)
    # Extract the filename without extension
    set filename = `basename "$root_file"`
    set filename_noext = "${filename:r}"

    # Construct the output JSON filename in the json folder
    set json_file = "/Cherenkov/ParticleGun/json/${filename_noext}.json"

    # Execute the root command with the appropriate parameters
    root -q DrcHit.cc+ DrcEvent.cc+ 'MakeDictionaries.C("'"${root_file}"'", "'"${json_file}"'")'
end
