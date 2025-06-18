# Depth Phase Array Analysis

Depth Phase Array Analysis is a seismic data processing workflow designed to detect and analyse depth phases such as P, pP, sP, S, and sS using adaptive, teleseismic ad-hoc arrays. It creates an initial earthquake catalogue, performs array processing (beamforming, vespagrams) for autmomatic phase detection and identification, these phases are used for event relocation in 3D using ISCloc, and has an optional feature to detect pmP phases for crustal thickness estimation.

## 📌 Description

This project automatically does the following:

- Creates an initial earthquake catalogue using [ObspyDMT](https://github.com/krischer/obspydmt)
- Downloads seismic waveform data (1 or 3 components)
- Applies array processing techniques with adaptive ad-hoc arrays
- Detects key depth phases (P, pP, sP, S, sS)
- Performs earthquake relocation using [ISCloc](https://www.isc.ac.uk/iscbulletin/iscloc/)
- Enables optional pmP detection for crustal thickness determination

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/AliceBlackwell/Depth_Phase_Array_Analysis.git
cd Depth_Phase_Array_Analysis/Scripts
```

Create conda environment and install packages:

```bash
conda create -n dpa-env python=3.10.11
conda activate dpa-env
pip install -r requirements.txt
```

Set up ISCloc:  

Go to [ISCloc](https://www.isc.ac.uk/iscbulletin/iscloc/) to download algorithm directory directly from the ISC, and move directory into Depth_Phase_Array_Analysis/Scripts  
Edit config file for ISCloc to suit needs:
```bash
nano ISClocRelease2.2.6/etc/iscloc/config.txt

# update_db = 0
# out_agency = AB  (the agency code your new relocation will be associated with, I use my initials)
# do_gridsearch = 0 (seeds on ISC location, 1 would complete a grid search centred on the median reported location and origin time)
# mindepthpha = 5
```

Compile ISCloc (including changing the Makefile to point towards conda packages) and edit .bashrc:

```bash
# From the Depth_Phase_Array_Analysis directory
mv compile_iscloc.sh ISClocRelease2.2.6/src  # (or src2.2.7 if available)
cd ISClocRelease2.2.6/src  # (or src2.2.7 if available)
source compile_iscloc.sh
```

## 🚀 Usage

Use the main.py wrapper to see an example of how to run the workflow steps.
To search for an initial earthquake catalogue to relocate, change the parameters in obspydmt.py.

```bash
python main.py n m
# n is the event index in the generated ObspyDMT catalogue, m is total events to process
# (leave blank for a single event use)
```

The workflow is fully described in ADD_PAPER_HERE.

## 🧾 Example Test Case

To try out the workflow using the example event and data provided in ObspyDMT_Events_test.  
This contains a pre-generated ObspyDMT event catalogue for the Mw 6.1 Peruvian event on 23rd May 2010, and a small selection of pre-downloaded 3-component seismic data.  

[Note: In a real use case, both the generation of the initial event catalogue and the data downloading are handled automatically by the workflow. These steps have been skipped in this example to streamline testing.]

From the Scripts/ directory, run the following command:
```bash
python main.py
```

This will: 
+ Output finalised phase lists
+ Relocate the event in 3D using ISCloc
+ Determine crustal thickness approximately overlying the event location

### 🔍 Comparing Results (your outputs) to Results_test (reference outputs)
After running the workflow with the example dataset (ObspyDMT_Events_test), you can verify the outputs by comparing your results to the provided reference results in the Results_test/ directory.  

Compare the following output files:

- **Final 3D relocated earthquake catalogue:**  
  `Final_3D_Catalogue.txt`

- **Final crustal thickness catalogue:**  
  `Final_cleaded_pmP_catalogue_5.9.txt`

- **1D relocated earthquake catalogue with ad-hoc array details**  
  *(not to be used for event location):*  
  `Final_1D_Catalogue_detailed.txt`

- **Final list of cleaned ad-hoc array phases:**  
  `20100523224651/Phase_list.txt`

### How to Compare
Use the diff command (or any file comparison tool) to inspect differences:
```bash
diff -r Results/file Results_test/file
```
