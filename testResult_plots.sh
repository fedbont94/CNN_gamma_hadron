#! /bin/sh

PYTHON=/hkfs/home/project/hk-project-pevradio/rn8463/virtual_env/bin/python3
SCRIPT=/home/hk-project-pevradio/rn8463/gamma_hadron/testResult_plots.py

$PYTHON $SCRIPT \
    --inputDir "/hkfs/work/workspace/scratch/rn8463-lv3_Simulations/" \
    --modelDir "/hkfs/work/workspace/scratch/rn8463-gamma_hadron/version2/" \
    --outputDir "/hkfs/home/project/hk-project-pevradio/rn8463/gamma_hadron/" \
    --year 2012 \
    --energyStart 4.0 \
    --energyEnd 7.0 
    
    