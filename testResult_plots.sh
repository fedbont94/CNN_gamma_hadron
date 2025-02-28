#! /bin/sh

PYTHON=/hkfs/home/project/hk-project-pevradio/rn8463/virtual_env/bin/python3
SCRIPT=/home/hk-project-pevradio/rn8463/gamma_hadron/testResult_plots.py

$PYTHON $SCRIPT \
    --inputDir "/hkfs/work/workspace/scratch/rn8463-lv3_Simulations/" \
    --modelDir "/hkfs/work/workspace/scratch/rn8463-gamma_hadron/version11/" \
    --outputDir "/hkfs/work/workspace/scratch/rn8463-gamma_hadron/version11/" \
    --year 2012 \
    --energyStart 5.0 \
    --energyEnd 6.5 
    
    