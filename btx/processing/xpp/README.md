# xppPumpProbe Analysis Pipeline

*xppPumpProbe* is an analysis pipeline for analyzing pump-probe experiments at XPP. The pipeline processes and analyzes data from pump-probe experiments, including loading raw data using smd, generating histograms, calculating Earth Mover's Distance (EMD), computing p-values, generating masks, and performing pump-probe analysis. The data loader may have to be redefined per-experiment, but the other components are reusable as is.

## Pipeline Structure

The *xppPumpProbe* analysis pipeline consists of several tasks that are executed sequentially. Each task is implemented as a separate class in the `xppmask.py` file. The tasks are:

1. **LoadData**: Loads raw data for pump-probe analysis.
2. **MakeHistogram**: Generates normalized histograms from loaded data.
3. **MeasureEMD**: Calculates Earth Mover's Distance (EMD) between histograms and a reference histogram.
4. **CalculatePValues**: Calculates p-values from EMD values and a null distribution.
5. **BuildPumpProbeMasks**: Generates binary masks based on p-values and ROI coordinates.
6. **PumpProbeAnalysis**: Performs pump-probe analysis using the generated masks and data.

## Configuration

todo
## Task Descriptions

### LoadData

The `LoadData` task is responsible for loading raw data for pump-probe analysis. It uses the `get_imgs_thresh` function from the `xpploader.py` file to load the data based on the specified run number, experiment number, ROI, energy filter, I0 threshold, IPM position filter, time bin, and time tool parameters.

### MakeHistogram

The `MakeHistogram` task generates normalized histograms from the loaded data. It uses the `calculate_histograms` function from the `histogram_analysis.py` file to create the histograms. The generated histograms are saved as a numpy file, and a summary file with histogram information is created.

### MeasureEMD

The `MeasureEMD` task calculates the Earth Mover's Distance (EMD) between histograms and a reference histogram. It uses the `calculate_emd_values` function from the `pvalues.py` file to compute the EMD values. The EMD values are saved as a numpy file, and a null distribution of EMD values is generated and saved as a separate numpy file. A summary file with EMD calculation information is also created.

### CalculatePValues

The `CalculatePValues` task calculates p-values from EMD values and a null distribution. It uses the `calculate_p_values` function from the `analysis_tasks.py` file to compute the p-values. The p-values are saved as a numpy file, and a summary file with p-value calculation information is generated.

### BuildPumpProbeMasks

The `BuildPumpProbeMasks` task generates binary signal and background masks based on the p-values, histograms, and ROI coordinates. It uses the `generate_signal_mask` and `generate_background_mask` functions from the `analysis_tasks.py` file to create the masks. The binary signal and background masks are saved as separate numpy files, and a summary file with mask generation information is produced.

### PumpProbeAnalysis

The `PumpProbeAnalysis` task performs the pump-probe analysis using the generated masks and loaded data. It groups images into stacks, generates intensity vs. delay curves, calculates p-values, plots the results, and saves the output files. The task produces a pump-probe plot, pump-probe curves data, a summary file, and a report file.

## Running the Pipeline

To run the *xppPumpProbe* analysis pipeline, follow these steps:

1. Prepare the YAML configuration file with the necessary parameters for each task.
2. Create an instance of each task class in the desired order, passing the configuration dictionary to the constructor.
3. Call the appropriate methods of each task instance to load data, perform computations, summarize results, report results, and save outputs. The output files and plots will be generated in the specified output directories.


## Testing

The *xppPumpProbe* pipeline includes a test script located in the `tests/test_xppmask.py` file. The test script contains individual test functions for each task and an integration test function (`test_pump_probe_analysis`) that runs all tasks in sequence. To run the tests, execute the `test_xppmask.py` script.
