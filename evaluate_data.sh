#! /bin/bash

ENV=/hkfs/home/project/hk-project-pevradio/rn8463/icetray/build/env-shell.sh
PYTHON=/hkfs/home/project/hk-project-pevradio/rn8463/virtual_env/bin/python3
SCRIPT=/home/hk-project-pevradio/rn8463/gamma_hadron/evaluate_data.py

# eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
$ENV $PYTHON $SCRIPT \
    --inputDir "/hkfs/work/workspace/scratch/rn8463-data_2012/hdf5/" \
    --modelDir "/hkfs/work/workspace/scratch/rn8463-gamma_hadron/version6/" \
    --outputDir "/hkfs/work/workspace/scratch/rn8463-data_2012/" \
    --year 2012     
    