# Depth Phase Array Analysis

Depth Phase Array Analysis is a seismic data processing workflow designed to detect and analyze depth phases such as P, pP, sP, S, and sS using ad-hoc arrays. It creates an initial earthquake catalogue, performs array processing for event relocation in 1D or 3D models using ISCloc, and enables advanced detection such as pmP phases for crustal thickness estimation.

## 📌 Description

This project automates the following:

- Creates an initial earthquake catalogue using [ObspyDMT](https://github.com/krischer/obspydmt)
- Downloads seismic waveform data (1 or 3 components)
- Applies array processing techniques with temporary or ad-hoc arrays
- Detects key depth phases (P, pP, sP, S, sS)
- Performs earthquake relocation using [ISCloc](https://www.isc.ac.uk/softiscloc/)
- Enables optional pmP detection for crustal structure analysis (e.g., crustal thickness)

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/depth-phase-array-analysis.git
cd depth-phase-array-analysis

pip install -r requirements.txt
