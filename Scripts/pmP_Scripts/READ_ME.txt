The is a directory with contents designed to identify pmP arrivals on previously generated vespagram data.
Written by Hana-Riia Allas and modified by Alice Blackwell.
[April 2025]

CONTENTS:
Classes_for_crthk_code.py                       
classes.py                                
crust1.py                              
Crustal_thickness_code_constantvel.py  
Failure_rate_pmP.py 
Finding_failure_rate.py   
Functions_for_crthk_code.py
pmp_submission.sh
Velocity_model_files

PURPOSE:
Classes_for_crthk_code.py      
-- defined class and methods to identify pmP arrivals from previuously constructed vespagram data in Crustal_thickness_code_constantvel.py.
                 
classes.py                                
-- classes defined and used during original vespagram construction, not directly used but needed to read in saved class results.

crust1.py                              
-- navigates the Crust 1.0 data (written by Crust 1.0 authors). **Has some file pathways at the top which need updating when directory is moved.**

Crustal_thickness_code_constantvel.py
-- Main script.
 
Failure_rate_pmP.py 
-- uses text file output by Finding_failure_rate.py to calculate failure rate of arrays when searching for pmP. Need to manually type in the number of pmP data points found.

Finding_failure_rate.py
--  outputs text file with event name and number of ad hoc arrays tested for pmP arrivals.

Functions_for_crthk_code.py
-- contains functions used in Crustal_thickness_code_constantvel.py.

pmp_submission.sh
-- example of HPC submission file.

Velocity_model_files
-- Crust 1.0 velocity files

