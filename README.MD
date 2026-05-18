# pyopenms-idfreeqc

ID-free QC metrics calculation for mass spectrometry mzML files using pyOpenMS.

This tool computes comprehensive quality control metrics from mzML files without requiring protein identification results. It outputs results in standard mzQC format with optional visualizations.

## Features

- **ID-free metrics**: Calculate QC metrics without peptide/protein identification
- **mzQC format**: Standard output format for mass spectrometry quality control
- **Comprehensive metrics**: 100+ metrics including RT duration, scan counts, TIC statistics, charge distributions, and more
- **Visualizations**: Optional heatmap generation for comparing multiple runs
- **Demo mode**: Built-in example data for testing and learning

## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Optional: install the latest pyOpenMS package, to enable latest features:
uv pip install --upgrade pyopenms --index-url https://pypi.cs.uni-tuebingen.de/simple/

# Install the package
uv pip install -e .
```

### Using pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

After installation, the tool is available as `pyopenms-idfreeqc`:

#### Basic Usage

```bash
# Process one or more mzML files
pyopenms-idfreeqc sample1.mzML sample2.mzML

# Process files using wildcards
pyopenms-idfreeqc data/*.mzML

# Specify custom output paths
pyopenms-idfreeqc sample.mzML -o my_results.mzQC -p my_plot.png
```

#### Demo Mode

Use built-in example data to test the tool:

```bash
# Download and process demo files
pyopenms-idfreeqc --demo --download-demo

# Use demo files (if already downloaded)
pyopenms-idfreeqc --demo
```

#### Options

- `--demo`: Use built-in demo mzML files
- `--download-demo`: Download demo files before processing
- `-o, --output PATH`: Output path for mzQC JSON; a TSV metrics table will also be saved next to it (default: `multi_run_qc.mzQC`)
- `-p, --plot PATH`: Output plot file path (default: `idfree_qc_plot.png`)
- `--no-plot`: Skip generating the heatmap visualization
- `--show-tables`: Print formatted metric tables to console
- `--show-json`: Print the full mzQC JSON output to console
- `--continue-on-error`:  Continue processing files even if an error occurs (still exits with error code)


#### Examples

```bash
# Process specific files with custom output
pyopenms-idfreeqc file1.mzML file2.mzML -o qc_metrics.json -p qc_heatmap.png

# Process demo files without plot
pyopenms-idfreeqc --demo --no-plot

# Show JSON output to console
pyopenms-idfreeqc sample.mzML --show-json

# Process files and show tables
pyopenms-idfreeqc *.mzML --show-tables
```

### As a Python Library

You can also use the tool programmatically in your Python scripts:

```python
from pyopenms_idfreeqc.calculate_metrics import calculate_metrics

# Process mzML files
json_output = calculate_metrics(
    mzml_files=["sample1.mzML", "sample2.mzML"],
    output_file="my_qc.json",
    generate_plot=True,
    plot_output="my_plot.png",
    show_tables=True,
    show_json=False
)

# The function returns the mzQC JSON as a string
print(json_output)
```

#### Library Function Parameters

- `mzml_files` (List[str]): List of paths to mzML files to process
- `output_file` (str, optional): Path for mzQC JSON output file
- `generate_plot` (bool): Whether to generate a heatmap visualization (default: True)
- `plot_output` (str): Path for the plot file (default: "idfree_qc_plot.png")
- `show_tables` (bool): Whether to print formatted tables to console (default: True)
- `show_json` (bool): Whether to print JSON to console (default: False)

**Returns**: mzQC JSON string

## Output Files

### mzQC JSON File

The mzQC (Mass Spectrometry Quality Control) file contains:
- Run metadata (instrument information, acquisition parameters)
- Quality metrics organized by category
- Controlled vocabulary terms (PSI-MS accessions)
- Multiple runs can be compared side-by-side

### Heatmap Visualization

The optional PNG heatmap shows:
- QC metrics as rows
- Different runs as columns
- Color-coded values (normalized per metric)
- Original values displayed as annotations

## Computed Metrics

The tool calculates over 100 quality control metrics including:

### Acquisition Metrics
- Chromatography duration
- Scan counts (MS1, MS2)
- Scan rates and frequencies
- RT ranges and distributions

### Signal Quality
- Total Ion Current (TIC) statistics
- Base peak intensities
- Signal stability (CV, jumps, falls)
- Empty scan counts

### MS2 Specific
- Precursor charge distributions
- Precursor intensity statistics
- Precursor m/z ranges

### Advanced Metrics
- Peak density quantiles
- RT-over-MS quantiles
- TIC quartile ratios
- FAIMS compensation voltages
- Chromatogram statistics
- Polarity statistics

See the source code for complete metric descriptions and PSI-MS accessions.

## Requirements

- Python >= 3.9
- pyOpenMS >= 3.4.0
- pymzqc >= 1.0.1
- click (for CLI)
- seaborn >= 0.13.2 (for visualizations)
- pandas, matplotlib, numpy (dependencies of above)

## Development

```bash
# Clone the repository
git clone <repository-url>
cd pyopenms-idfreeqc

# Install with uv in development mode
uv venv
source .venv/bin/activate
uv pip install -e .

# Run tests
pyopenms-idfreeqc --demo --download-demo
```

## License

See LICENSE file for details.

## Citation

If you use this tool in your research, please cite the relevant publications.
