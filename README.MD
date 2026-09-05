# rawQC

Identification-free quality control for DDA and DIA mass spectrometry data using pyOpenMS. rawQC reads mzML files and native Bruker TDF `.d` datasets, with conditional support for Thermo `.raw` files. It writes mzQC and TSV results and can generate a heatmap for comparing runs.

## Features

- **ID-free metrics**: Calculate QC metrics without peptide or protein identifications.
- **DDA and DIA metrics**: Shared acquisition and signal statistics, with separate precursor metrics for DDA and isolation-window metrics for DIA.
- **mzQC output**: PSI-MS accessions where applicable, with custom rawQC terms for additional metrics.
- **Visualizations**: Optional heatmaps for comparing multiple runs.
- **Demo mode**: Built-in example mzML data.

## Installation

Python 3.10 or newer is required. The project pins **`pyopenms==3.6.0.dev20260905`** from the [OpenMS nightly package index](https://pypi.openms.de/simple/) to provide the native Bruker reader. This is a specific nightly version, so updates require changing the pin and lockfile together.

### Using uv (recommended)

From a checkout of this repository, install the project and its locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --no-dev
uv run rawQC --help
```

`pyproject.toml` assigns pyOpenMS to the explicit `openms-nightly` index; other dependencies use the default package index. Omit `--no-dev` to include development dependencies.

The commands below assume the environment is active. Alternatively, prefix each command with `uv run`, for example `uv run rawQC sample.mzML`.

### Using pip

pip does not read `[tool.uv.sources]`, so install the pinned nightly explicitly before installing rawQC:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

python -m pip install "pyopenms==3.6.0.dev20260905" \
    --index-url https://pypi.openms.de/simple/ \
    --extra-index-url https://pypi.org/simple/
python -m pip install -e .
```

## Supported Inputs

| Input | Support |
| --- | --- |
| `.mzML` file | DDA and DIA through `MzMLFile`. Use `--acquisition-mode dia` for DIA files without an explicit acquisition annotation. |
| Bruker `.d` directory | Native TDF datasets containing both `analysis.tdf` and `analysis.tdf_bin`, through `BrukerTimsFile`. TSF and other `.d` formats are unsupported. |
| Thermo `.raw` file | Attempted through `FileHandler`; requires a pyOpenMS build with `WITH_THERMO_RAW` and its .NET runtime. The pinned Linux and Windows wheels disable this reader; convert to mzML on those platforms. macOS native loading has not been verified and also depends on the build and runtime. |

The Bruker reader's default AUTO export preserves DIA isolation windows and ion-mobility data. rawQC does not request cross-frame averaging, denoising, or optional centroiding. The native reader can omit empty frame/window spectra. For native DIA input, rawQC also reads the acquisition metadata and reports expected spectrum counts and differences from the loaded counts. These differences describe reader omissions, not missing acquisitions; loaded empty-scan counts cannot recover omitted spectra.

## Usage

### Command Line Interface

```bash
# Process one or more mzML files
rawQC sample1.mzML sample2.mzML

# Explicitly select DIA metrics for an unannotated mzML file
rawQC dia_sample.mzML --acquisition-mode dia

# Process native Bruker datasets; DIA native IDs enable automatic selection
rawQC sample1.d sample2.d

# Thermo DIA input, only with a build supporting native Thermo loading
rawQC dia_sample.raw --acquisition-mode dia

# Specify output paths
rawQC data/*.mzML -o my_results.mzQC -p my_plot.png

# Download and process the demo data without a plot
rawQC --demo --download-demo --no-plot
```

### Acquisition Mode

`--acquisition-mode auto` is the default. It selects DIA when the experiment is explicitly annotated as DIA, or when all MS2 spectra have DIA annotations or Bruker DIA `windowGroup=` native IDs. Otherwise it retains the existing DDA behavior. Isolation width or repeated precursor m/z alone does not trigger DIA selection.

Use `--acquisition-mode dia` for known DIA data without those annotations, including converted mzML files. Use `--acquisition-mode dda` to select DDA explicitly. A mixture of DIA-marked and unannotated MS2 spectra requires an explicit mode or separate inputs; rawQC does not automatically partition mixed acquisitions. An explicit mode applies to all files in that command.

### Options

- `--acquisition-mode [auto|dda|dia]`: Select acquisition-specific metrics (default: `auto`).
- `--demo`: Use built-in demo mzML files.
- `--download-demo`: Download demo files before processing.
- `-o, --output PATH`: mzQC JSON output; a TSV table is saved alongside it (default: `multi_run_qc.mzQC`).
- `-p, --plot PATH`: Heatmap output (default: `idfree_qc_plot.png`).
- `--no-plot`: Skip the heatmap.
- `--show-tables`: Print metric tables.
- `--show-json`: Print the mzQC JSON.
- `-c, --cmap TEXT`: Heatmap colormap (default: `RdBu_r`).
- `--continue-on-error`: Process remaining inputs after an error; the command still exits with a nonzero status.

### As a Python Library

```python
from rawQC.calculate_metrics import calculate_metrics

error_occurred = calculate_metrics(
    mzml_files=["dia_sample.mzML", "bruker_sample.d"],
    acquisition_mode="dia",
    output_file="my_qc.mzQC",
    generate_plot=True,
    plot_output="my_plot.png",
    show_tables=True,
)
```

`calculate_metrics()` writes the output files and returns an error flag: `False` when processing succeeds, or `True` if an input fails while `continue_on_error=True`. With the default `continue_on_error=False`, processing errors raise an exception. The return value is not the mzQC JSON.

The legacy `mzml_files` argument now accepts all supported input paths. Set `output_file=None` to skip mzQC and TSV file output. Table and JSON console output are disabled by default.

For an in-memory metric dictionary:

```python
from rawQC.input import load_experiment
from rawQC.calculate_metrics import compute_qc_metrics

experiment = load_experiment("dia_sample.mzML")
metrics = compute_qc_metrics(experiment, acquisition_mode="dia")
```

## Output Files

- **mzQC JSON**: Run and instrument metadata, the selected acquisition mode, input-format metadata, and QC metrics. DIA-specific metrics use custom rawQC terms.
- **TSV table**: Metrics for all runs, saved next to the mzQC file.
- **Optional PNG heatmap**: Numeric metrics normalized across runs, with the original values shown as annotations.

## Computed Metrics

### Shared Acquisition and Signal Metrics

- Chromatography duration, spectrum counts, scan rates, frequencies, and retention-time ranges.
- TIC statistics, base-peak intensities, signal variability, and loaded empty-scan counts.
- Peak-density quantiles, RT-over-MS quantiles, and TIC quartile statistics.
- FAIMS compensation voltages, chromatogram statistics, polarity, peak types, and instrument information.

### DDA Metrics

- Selected-precursor charge distributions and charge ratios.
- Precursor intensity quantiles, mean, sample standard deviation, and intensity extent.
- Precursor m/z range and median.
- Count of zero/unrecorded precursor intensities replaced by the MS2 TIC under the existing QuaMeter fallback.

### DIA Metrics

- Number of loaded MS2 spectra and counts of missing, invalid, or ambiguous isolation metadata.
- Distinct isolation windows, their m/z envelope, and width range.
- A window table with m/z bounds, available ion-mobility or FAIMS metadata, scan counts, loaded empty-scan counts, invalid-intensity counts, and fragment TIC sums and medians.
- MS2 fragment TIC area, summing spectra at identical retention times before integration so simultaneous diaPASEF windows do not introduce an ordering artifact. At least two distinct valid retention times are required.
- MS1-bounded survey-cycle counts, duration median and CV, and MS2/window counts per interval.
- For native Bruker DIA, acquisition-metadata spectrum counts and reader-omission counts, when available.

Window bounds include both isolation offsets. Windows with different mobility metadata remain separate. Spectra with multiple precursors are reported as ambiguous and excluded from the window table, avoiding duplicate assignment of fragment signal. Fragment TIC is not normalized by window width. DIA output excludes the DDA charge and selected-precursor metrics, including the precursor-intensity TIC fallback.

Cycle summaries use acquisition-order intervals between consecutive MS1 spectra with valid increasing retention times and enclosed MS2 spectra. Leading and trailing unbounded MS2 spectra are excluded from cycle summaries but remain eligible for window statistics. These intervals do not establish completeness of a DIA cycle or a FAIMS supercycle. Without usable boundaries, duration and per-cycle summaries are unavailable. The isolation m/z envelope likewise does not imply gap-free coverage. Unavailable values are written as JSON `null`.

See the metric descriptions in the source for the exact definitions and accessions.

## Development

```bash
git clone https://github.com/bigbio/rawQC.git
cd rawQC
uv sync --locked
uv run pytest
```

Acquisition-specific code is separated into `src/rawQC/dda.py` and `src/rawQC/dia.py`. `input.py` handles readers and acquisition-mode selection; `calculate_metrics.py` contains shared metrics, orchestration, output, and the CLI. Small shared helpers live in `spectrum_utils.py`. Existing DDA function imports from `rawQC.calculate_metrics` remain available.

## License

See LICENSE file for details.

## Citation

If you use this tool in your research, please cite the relevant publications.
