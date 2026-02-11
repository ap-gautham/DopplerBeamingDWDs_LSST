# Presaging Doppler beaming discoveries of double white dwarfs during the Rubin LSST era
These are the files used to produce the results of the paper "Presaging Doppler beaming discoveries of double white dwarfs during the Rubin LSST era"

You will need to download the LSST baseline observation database file 'baseline_v5.0.0_10yrs.db' and include in the the data folder.

# SeBa DWD simulation
We used a modified version of SeBa. We did not change any of the physics and only modified the output writing of the package. We can provide the steps to exactly modify the file upon request, but do not provide the modified package.

The notebooks were written to be run on multiple nodes on the Rocksfish cluster. The notebooks will need to be modified to rerun the DWD simulation on your setup.

-- 03_SeBa/01_GenerateSimulateDWDFormation.ipynb creates the coordinates and distances for binaries in the galaxy and creates shell scripts to simulate the binary evolution
---- To generate SeBa binaries, run the created shell scripts. 
-- 03_SeBa/02_Collect_GenerateSimulateDWDFormation.ipynb collects the simulated result and creates the collection of DWD binaries
-- 03_SeBa/03_Analyze_GenerateSimulateDWDFormation.ipynb distributes the simulated binaries across the galaxy and performs mock photometry

The final result "GalacticDWDs.ecsv" can then be used to simulate the LSST lightcurves using 01_LSST_LightCurveSimulation.py


This repositry is under construction and will be updated over time.
