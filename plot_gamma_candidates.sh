#! /bin/bash

ENV=/hkfs/home/project/hk-project-pevradio/rn8463/icetray/build/env-shell.sh
PYTHON=/hkfs/home/project/hk-project-pevradio/rn8463/virtual_env/bin/python3
SCRIPT=/home/hk-project-pevradio/rn8463/gamma_hadron/plot_gamma_candidates.py

# eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
$ENV $PYTHON $SCRIPT \
    --inputDir "/hkfs/work/workspace/scratch/rn8463-data_2012/gamma_candidates/" \
    --outputDir "/hkfs/work/workspace/scratch/rn8463-data_2012/" \
    --year 2012     
    