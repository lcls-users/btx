import os
import numpy as np
import h5py
from xppmask import *
from xppmask import MakeHistogram
from xppmask import LoadData
import yaml

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def test_load_data(output_dir, config):


    load_data = LoadData(config)
    load_data.load_data()
    load_data.save_data()
    load_data.summarize()

    output_file = os.path.join(output_dir, 'load_data', f"{config['setup']['exp']}_{config['setup']['run']}", f"{config['setup']['exp']}_run{config['setup']['run']}_data.npz")
    assert os.path.exists(output_file), f"Output file {output_file} does not exist"

    # Check if binned_delays are saved in the output file
    with np.load(output_file) as data:
        assert 'binned_delays' in data, "binned_delays not found in the output file"
        assert data['binned_delays'].shape == data['laser_delays'].shape, "Shape mismatch between binned_delays and laser_delays"

    return output_file

def test_make_histogram(data_file, output_dir, config):


    make_histogram = MakeHistogram(config)
    make_histogram.load_data()
    make_histogram.generate_histograms()
    make_histogram.summarize()

    histogram_file = make_histogram.save_histograms()

    assert os.path.exists(histogram_file), f"Histogram file {histogram_file} does not exist"
    assert histogram_file == os.path.join(output_dir, 'make_histogram', f"{config['setup']['exp']}_{config['setup']['run']}", "histograms.npy")

    # Save the histogram image
    make_histogram.save_histogram_image()

    # Return the path to the histogram file and the summary
    return histogram_file, make_histogram.histogram_summary

def test_make_histogram_from_npz(npz_file, output_dir, config):


    # Create and run the MakeHistogram instance
    make_histogram = MakeHistogram(config)
    
    # Load data directly from the npz file
    with np.load(npz_file) as data:
        make_histogram.data = data['data']
    
    make_histogram.generate_histograms()
    make_histogram.summarize()

    # Save the histograms using the save_histograms() method
    histogram_file = make_histogram.save_histograms()

    # Check if the histogram file exists
    assert os.path.exists(histogram_file), f"Histogram file {histogram_file} does not exist"

    # Save the histogram image
    make_histogram.save_histogram_image()

    # Return the path to the histogram file and the summary
    return histogram_file, make_histogram.histogram_summary

def test_measure_emd(histogram_file, output_dir, config):


    emd_task = MeasureEMD(config, histogram_file)
    emd_task.load_histograms()
    emd_task.calculate_emd()
    emd_task.summarize()

    report_path = os.path.join(emd_task.output_dir, 'report.txt')
    emd_values_path = os.path.join(emd_task.output_dir, "emd_values.npy")
    emd_null_dist_path = os.path.join(emd_task.output_dir, "emd_null_dist.npy")

    assert os.path.exists(report_path), f"Report file {report_path} does not exist"
    assert os.path.exists(emd_values_path), f"EMD values file {emd_values_path} does not exist"
    assert os.path.exists(emd_null_dist_path), f"EMD null distribution file {emd_null_dist_path} does not exist"

    assert emd_values_path == os.path.join(output_dir, 'measure_emd', f"{config['setup']['exp']}_{config['setup']['run']}", "emd_values.npy")
    assert emd_null_dist_path == os.path.join(output_dir, 'measure_emd', f"{config['setup']['exp']}_{config['setup']['run']}", "emd_null_dist.npy")

    return os.path.join(emd_task.output_dir, "emd_values.npy")


def test_calculate_p_values(emd_file, output_dir, config):


    pval_task = CalculatePValues(config)
    pval_task.load_data()
    pval_task.calculate_p_values()
    pval_task.summarize()

    report_path = os.path.join(pval_task.output_dir, 'pvalues_report.txt')
    p_values_path = os.path.join(pval_task.output_dir, "p_values.npy")

    assert os.path.exists(report_path), f"Report file {report_path} does not exist"
    assert os.path.exists(p_values_path), f"P-values file {p_values_path} does not exist"

    assert p_values_path == os.path.join(output_dir, 'calculate_p_values', f"{config['setup']['exp']}_{config['setup']['run']}", "p_values.npy")

    return os.path.join(pval_task.output_dir, "p_values.npy")

def test_build_pump_probe_masks(p_values_file, histograms_file, output_dir, config):


    mask_task = BuildPumpProbeMasks(config)
    mask_task.load_histograms()
    mask_task.load_p_values()
    mask_task.generate_masks()
    mask_task.summarize()

    report_path = os.path.join(mask_task.output_dir, 'masks_report.txt')
    signal_mask_path = os.path.join(mask_task.output_dir, "signal_mask.npy")
    bg_mask_path = os.path.join(mask_task.output_dir, "bg_mask.npy")

    assert os.path.exists(report_path), f"Report file {report_path} does not exist"
    assert os.path.exists(signal_mask_path), f"Signal mask file {signal_mask_path} does not exist"
    assert os.path.exists(bg_mask_path), f"Background mask file {bg_mask_path} does not exist"

    assert signal_mask_path == os.path.join(output_dir, 'build_pump_probe_masks', f"{config['setup']['exp']}_{config['setup']['run']}", "signal_mask.npy")
    assert bg_mask_path == os.path.join(output_dir, 'build_pump_probe_masks', f"{config['setup']['exp']}_{config['setup']['run']}", "bg_mask.npy")

    return signal_mask_path, bg_mask_path


def test_pump_probe_analysis(config, data_file, signal_mask_file, bg_mask_file):
    # Load the output data from LoadData and BuildPumpProbeMasks
    data = np.load(data_file, allow_pickle=True)
    signal_mask = np.load(signal_mask_file)
    bg_mask = np.load(bg_mask_file)

    # Initialize and run PumpProbeAnalysis
    analysis = PumpProbeAnalysis(config)
    analysis.run(data['data'], data['binned_delays'], data['I0'], data['laser_on_mask'], data['laser_off_mask'], signal_mask, bg_mask)

    # Check that output files were generated
    assert os.path.exists(analysis.output_dir), f"Output directory {analysis.output_dir} was not created"

    expected_files = ['pump_probe_plot.png', 'pump_probe_curves.npz', 'pump_probe_summary.txt', 'pump_probe_report.txt']
    for file in expected_files:
        file_path = os.path.join(analysis.output_dir, file)
        assert os.path.isfile(file_path), f"Expected output file {file_path} was not generated"

    # Load and check pump_probe_curves.npz
    curves = np.load(os.path.join(analysis.output_dir, 'pump_probe_curves.npz'))
    expected_keys = ['delays', 'signals_on', 'std_devs_on', 'signals_off', 'std_devs_off', 'p_values']
    for key in expected_keys:
        assert key in curves, f"Expected key {key} not found in pump_probe_curves.npz"
        assert curves[key].shape == (len(analysis.delays),), f"Unexpected shape for {key} in pump_probe_curves.npz"

    # Check pump_probe_summary.txt
    with open(os.path.join(analysis.output_dir, 'pump_probe_summary.txt'), 'r') as f:
        summary = f.read()
    assert f"Pump-probe analysis for {config['setup']['exp']} run {config['setup']['run']}:" in summary
    assert f"Number of delays: {len(analysis.delays)}" in summary
    assert f"Results saved to: {analysis.output_dir}" in summary

    # Check pump_probe_report.txt
    with open(os.path.join(analysis.output_dir, 'pump_probe_report.txt'), 'r') as f:
        report = f.read()
    assert f"Data loaded from: {config['setup']['root_dir']}" in report

    signal_mask_path = os.path.join(
        config['setup']['output_dir'], 
        'masks', 
        f"{config['setup']['exp']}_{config['setup']['run']}", 
        'signal_mask.npy'
    )
    expected_message = f"Signal mask loaded from: {signal_mask_path}"
    assert expected_message in report

#    bg_mask_path = os.path.join(
#        config['setup']['output_dir'], 
#        'masks', 
#        f"{config['setup']['exp']}_{config['setup']['run']}", 
#        'bg_mask.npy'
#    )
#    expected_message = f"Background mask loaded from: {bg_mask_path}"
#    assert expected_message in report

    assert "Laser on signals:" in report
    assert "Laser off std devs:" in report
    assert "p-values:" in report

# test_xppmask.py

if __name__ == '__main__':
    # Load the configuration from xppmask.yaml
    config_path = 'xppmask.yaml'
    test_config = load_yaml_config(config_path)
    test_config['setup']['run'] = 195

    # Run LoadData
    data_file = test_load_data(test_config['setup']['output_dir'], test_config)

    # Run MakeHistogram with LoadData output
    hfile_from_load_data, summary_from_load_data = test_make_histogram(data_file, test_config['setup']['output_dir'], test_config)
    print("MakeHistogram with LoadData output:")
    print(summary_from_load_data)

    # Run MeasureEMD and save the output
    emd_file = test_measure_emd(hfile_from_load_data, test_config['setup']['output_dir'], test_config)

    # Run CalculatePValues and save the output
    p_values_file = test_calculate_p_values(emd_file, test_config['setup']['output_dir'], test_config)

    # Run BuildPumpProbeMasks
    signal_mask_file, bg_mask_file = test_build_pump_probe_masks(p_values_file, hfile_from_load_data, test_config['setup']['output_dir'], test_config)

    try:
        test_pump_probe_analysis(test_config, data_file, signal_mask_file, bg_mask_file)
        print("PumpProbeAnalysis test passed")
    except AssertionError as e:
        print(f"PumpProbeAnalysis test failed: {str(e)}")
