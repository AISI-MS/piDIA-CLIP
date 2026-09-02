# πDIA-CLIP: Efficient Identification of Highly Heterogeneous Proteomics Data via a Generalized Zero-Shot Framework

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.09.704949-brightgreen)](https://doi.org/10.64898/2026.02.09.704949)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.18863866-blue)](https://zenodo.org/records/21850269)

## Description

**πDIA-CLIP** (Data-Independent Acquisition with Contrastive Learning Integrated Proteomics) is a generalized framework for zero-shot DIA-MS analysis. The prefix **π** denotes the [π-HuB project](https://doi.org/10.1038/s41586-024-08249-7), a global initiative for proteomics research.

If you use πDIA-CLIP in your work, please cite:

> Liao Y, Li Y, Xiao Z, Miao C, Zhao X, Zhang Y, Wen H, E W, Chang C, Zhang W. πDIA-CLIP: efficient identification of highly heterogeneous proteomics data via a generalized zero-shot framework. *bioRxiv* (2026). https://doi.org/10.64898/2026.02.09.704949

## Key Features

- **Zero-shot inference-only architecture** — no run-specific semi-supervised re-training; plug-and-play PSM re-scoring after standard RT calibration and XIC extraction
- **Cross-modal contrastive learning** — aligns peptide sequences and multi-dimensional XIC signals in a shared latent space via a transformer-based sequence encoder and a specialized spectral encoder
- **Hybrid encoder–decoder scoring** — combines aligned latent features with co-elution statistics for calibrated PSM scores and fragment-based quantification
- **Broad applicability** — validated on bulk DIA, multi-species mixtures, metaproteomics, spatial proteomics, and single-cell proteomics
- **Hardware-agnostic efficiency** — CPU and GPU inference; substantially faster than existing tools

## Workflow

πDIA-CLIP integrates into the standard peptide-centric DIA-MS workflow at the **PSM re-scoring** stage. Retention-time calibration and peak-group generation can be performed by mainstream tools (e.g., DIA-NN, MaxDIA, Spectronaut). πDIA-CLIP then extracts precursor and fragment XICs and performs zero-shot re-scoring, FDR estimation, and quantification.

```
DIA-MS raw data → RT calibration / XIC extraction (DIA-NN or compatible tools)
                → πDIA-CLIP zero-shot PSM re-scoring & quantification
                → Target-decoy FDR filtering → Identification & quantification tables
```

## Software

| Component | Description |
|-----------|-------------|
| **Command-line inference** | This repository (`scripts/infer_script.py`) |
| **GUI application** | Pre-built Windows application with a graphical interface — [GitHub Releases](https://github.com/AISI-MS/piDIA-CLIP/releases) |
| **Agent system** | Browser-based assistant for interactive analysis — [Primary access](http://yxxb1548675.bohrium.tech:50002/) |

Training data, benchmark results, and example files (mzML, spectral library, model weights) are available on [Zenodo](https://doi.org/10.5281/zenodo.18863866).

### GUI application

Users who prefer not to configure the Python environment or run the command-line workflow can download the packaged GUI installer from [GitHub Releases](https://github.com/AISI-MS/piDIA-CLIP/releases). The installer includes the inference runtime, DIA-NN runtime, and checkpoint model files.

The GUI is intended for 64-bit Windows 10/11. At least 20 GB of free disk space is recommended for installation and output files. Install the application to a path containing only letters, numbers, or underscores; avoid spaces and non-ASCII or special characters.

To run an analysis:

1. Open the application and create a new **πDIA-CLIP** task.
2. Select a spectral library (`.parquet`, `.tsv`, or `.fasta`) and add one or more mzML files.
3. Select the DIA-NN backbone, inference mode, device, thread count, batch size, and FDR threshold. DIA-NN 2.0 with a Parquet library is recommended for the current workflow.
4. Confirm that the validation panel reports **Ready to run**, then click **Run**.
5. Monitor the terminal and progress panels, and open the configured output directory after the task completes.

The complete **πDIA-CLIP GUI Software User Manual** is provided with the release documentation.

### Agent system

The πDIA-CLIP Agent provides a browser-based conversational interface for data preparation, inference, result download, and visualization. Use the primary address when available; the fallback nodes have fewer CPU and memory resources and may run more slowly.

- **Primary:** [http://yxxb1548675.bohrium.tech:50002/](http://yxxb1548675.bohrium.tech:50002/)
- **Fallback 1:** [http://qjqj1548978.bohrium.tech:50002/](http://qjqj1548978.bohrium.tech:50002/)
- **Fallback 2:** [http://rlas1548980.bohrium.tech:50002/](http://rlas1548980.bohrium.tech:50002/)

Open one of the addresses in a browser, upload the required files, and describe the requested workflow in the message box. Common workflows include:

- **mzML + spectral library:** run direct identification using an uploaded `.tsv` or `.parquet` library.
- **RAW + FASTA:** convert RAW data to mzML, generate a spectral library, and run inference.
- **FASTA only:** generate a reusable spectral library.
- **Identification results + mzML:** open previously generated results in the interactive viewer without rerunning inference.

For the DIA-NN 2.0 workflow with a Parquet library, select **Deep** mode for routine analyses or **Precise** mode when stricter false-positive control is preferred. Progress and logs are displayed in the chat, and generated files can be downloaded when processing finishes. Results from the current workflow can also be opened in the web viewer for chromatogram, MS/MS spectrum, and protein-coverage inspection.

The complete **πDIA-CLIP Agent User Manual** is provided with the release documentation.

---

## Requirements

- Python 3.x (Python 3.12 is recommended)
- CUDA is recommended for faster inference (optional; CPU is supported)
- DIA-NN 2.0, downloaded separately. The DIA-NN binaries and runtime libraries are not included in this repository because some of them exceed GitHub's file-size limit.

Download DIA-NN 2.0 from its official distribution and copy the complete runtime files into `scripts/diann-2.0/`. This directory must contain the executable for your platform (`diann.exe` on Windows or `diann-linux` on Linux) together with all required companion files, including the distributed `.dll` and `.so` libraries. Do not copy only the executable.

Ensure that the executable has execute permission. On Linux, for example:

```bash
chmod +x scripts/diann-2.0/diann-linux
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd DIA-CLIP
```

### 2. Create a virtual environment (conda is recommended)

```bash
conda create -n dia-clip python=3.12
conda activate dia-clip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes PyTorch, NumPy, pandas, SciPy, lxml, and PyArrow. PyArrow is required for reading and writing the DIA-NN and DIA-CLIP Parquet files.

Typical installation time is approximately 5 minutes on a standard desktop computer, excluding the time required to download the example data and model checkpoints. Actual installation time may vary depending on network speed.

---

## Configuration

Inference is driven by a **config file**. Use `configs/infer.config` as a reference.

### Example config

Create or edit a `.config` file in the project root or under `configs/`:

```ini
[General]
device = cpu              # Compute device: cpu or cuda
dtype = float             # Model precision: float or half
threads = 8               # Number of worker threads/processes
out_dir = ./results       # Output directory; created when needed

[Preprocess]
diann_dir = ./diann-2.0/ # Directory containing the complete DIA-NN runtime
library_path = library.parquet
mzml_path = 1.mzML;2.mzML # Separate multiple mzML paths with ;

[Database]
batch_size = 1024         # Inference batch size

[Infer]
checkpoint_path = model.pt
fdr = 0.01                # FDR threshold (0.01 means 1%)
quant_topk = 6            # Top fragment ions used for quantification

[Model]
# These values must match the checkpoint. See configs/infer.config for all options.
rt_dim = 12
ion_num = 10
hidden_layer = 1024
d_model = 512
n_head = 64
```

### Required options

| Option | Description |
|--------|-------------|
| `out_dir` | Output directory; it is created when needed |
| `diann_dir` | Directory containing the complete DIA-NN runtime, including its executable and companion libraries |
| `library_path` | Path to the DIA-NN spectral library (Parquet is supported) |
| `mzml_path` | One or more mzML file paths, separated by `;` |
| `checkpoint_path` | Path to the pretrained model checkpoint (`.pt`) |

The options in `[Model]` define the model architecture and must match the checkpoint. Unless you are using a different compatible checkpoint, keep the values provided in `configs/infer.config`.

---

## Usage

### Example data

Example data, including mzML files, a DIA-NN-generated spectral library, and model weights (`.pt`), is available from [Zenodo](https://zenodo.org/records/21850269).

### 1. Prepare data

- **mzML files**: DIA mass spectrometry raw data (`.mzML` is supported).
- **Spectral library**: A peptide spectral library generated by DIA-NN (Parquet is supported).
- **Model weights**: A `.pt` checkpoint.
- **DIA-NN**: Download DIA-NN 2.0 separately and copy its executable and all distributed runtime libraries (`.dll`, `.so`, and other companion files) into `scripts/diann-2.0/`. Make sure `diann.exe` or `diann-linux` is executable.

### 2. Edit the config

Copy and modify `configs/infer.config` (or your own config), and set the paths, for example:

```ini
[General]
device = cuda
out_dir = ./results

[Preprocess]
diann_dir = ./diann-2.0/
library_path = /path/to/your/library.parquet
mzml_path = /path/to/sample1.mzML;/path/to/sample2.mzML

[Infer]
checkpoint_path = /path/to/model.pt
```

Keep the complete `[Model]` section from `configs/infer.config` in your configuration.

### 3. Run inference

From the project **root directory**:

```bash
cd DIA-CLIP
python scripts/infer_script.py --config configs/infer.config
```

The `--config` argument is required. Before inference, DIA-CLIP runs DIA-NN and writes its intermediate files to `<out_dir>/diann/`.

### Demo runtime and expected result

When processing the 0.5 h HeLa demo dataset on a standard desktop computer using all available CPU cores, the complete workflow, including DIA-NN processing and DIA-CLIP inference, is expected to finish within approximately 20 minutes. Set `threads` to the number of logical CPU cores available on the computer to use all cores.

At `fdr = 0.01`, the FDR-filtered output is expected to contain approximately 55,000–61,000 identifications. The exact number depends on the checkpoint used and may also be affected by the configured FDR threshold.

### 4. Output

For each mzML file, three final result files are written under `out_dir`:

- **`<basename>.diaclip.all.parquet`**: All peptide inference and quantification results.
- **`<basename>.diaclip.fdr.parquet`**: Results passing the configured FDR threshold, including DIA-NN metadata.
- **`<basename>.diaclip.fdr.tsv`**: TSV version of the FDR-filtered results.

DIA-NN intermediate files, including `all_report.parquet` and `all_lib.parquet`, are written under `<out_dir>/diann/`. When multiple mzML files are processed in one run, these intermediate filenames are reused, while the final DIA-CLIP filenames remain sample-specific.

Example: with `mzml_path = sample1.mzML` and `out_dir = ./results` you get:

- `results/sample1.diaclip.all.parquet`
- `results/sample1.diaclip.fdr.parquet`
- `results/sample1.diaclip.fdr.tsv`

## Citation

```bibtex
@article{Liao2026piDIACLIP,
  title   = {πDIA-CLIP: efficient identification of highly heterogeneous proteomics data via a generalized zero-shot framework},
  author  = {Liao, Yucheng and Li, Yongge and Xiao, ZeXu and Miao, ChenChen and Zhao, Xingpu and Zhang, Yuanyuan and Wen, Han and E, Weinan and Chang, Cheng and Zhang, Weijie},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2026.02.09.704949}
}
```

## Contact

For questions or collaboration inquiries, please contact the corresponding authors:

- Cheng Chang — changcheng@ncpsb.org.cn
- Weijie Zhang — zhangwj@aisi.ac.cn
