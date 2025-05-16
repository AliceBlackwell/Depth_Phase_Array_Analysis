# Depth Phase Array Analysis

Depth Phase Array Analysis is a seismic data processing workflow designed to detect and analyse depth phases such as P, pP, sP, S, and sS using adaptive, teleseismic ad-hoc arrays. It creates an initial earthquake catalogue, performs array processing (beamforming, vespagrams) for autmomatic phase detection and identification, these phases are used for event relocation in 3D using ISCloc, and has an optional feature to detect pmP phases for crustal thickness estimation.

## 📌 Description

This project automates the following:

- Creates an initial earthquake catalogue using [ObspyDMT](https://github.com/krischer/obspydmt)
- Downloads seismic waveform data (1 or 3 components)
- Applies array processing techniques with adaptive ad-hoc arrays
- Detects key depth phases (P, pP, sP, S, sS)
- Performs earthquake relocation using [ISCloc](https://www.isc.ac.uk/iscbulletin/iscloc/)
- Enables optional pmP detection for crustal thickness determination

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/depth-phase-array-analysis.git
cd depth-phase-array-analysis

conda create -n dpa-env python=3.10.11
conda activate dpa-env
pip install -r requirements.txt
```

## 🚀 Usage

Use the main.py wrapper to see an example of how to run the workflow steps.
To search for an initial earthquake catalogue to relocate, change the parameters in obspydmt.py.

The workflow is fully described in ADD_PAPER_HERE.
