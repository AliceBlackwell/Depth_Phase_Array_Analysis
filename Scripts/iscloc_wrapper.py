# Python script written by AEB to loop through input files for ISCloc (January 2024)

import os
import subprocess
from classes import Earthquake_event

def run_iscloc(inputs_dir, station_list_dir, outputs_dir, all_events=True, catalogue=False, event=False):
    if all_events == True:
        for file in os.listdir(inputs_dir):          
            if '.in' in str(file):  # augmented ISC phases (plus ad-hoc array phases)
            
                if os.path.isfile(outputs_dir + str(file)[:-2] + 'out'):
                    continue

                # Infile
                print(file) 
                infile = inputs_dir + str(file)

                # Make outfile name
                event = str(file)[:-3]
                outfile = outputs_dir + event + '.out'
                print(event, outfile)

                command = 'echo "update_db=0 isf_stafile=' + station_list_dir+ str(event) +' isf_infile=' + infile + ' isf_outfile=' + str(outfile) + '" | ./iscloc_nodb isf >' + outputs_dir + str(event) + '.log'
                print(command)
                os.system(command)

            if '.dat' in str(file):  # original ISC phases
                
                if os.path.isfile(outputs_dir + str(file)[5:-4] + '.ISFout'):
                    continue
                
                # Infile
                print(file) 
                infile = inputs_dir + str(file)

                # Make outfile name
                event = str(file)[5:-4]
                outfile = outputs_dir + event + '.ISFout'
                print(event, outfile)

                command = 'echo "update_db=0 isf_stafile=' + station_list_dir+ str(event) +' isf_infile=' + infile + ' isf_outfile=' + str(outfile) + '" | ./iscloc_nodb isf >' + outputs_dir + str(event) + '.log'
                print(command)
                os.system(command)
                
    if all_events == False and event != False:
    
        # Select event from catalogue and define attributes
        event = Earthquake_event(catalogue[int(event)-1])
        event.define_event()
        ev_id = event.event_id
        
        for file in os.listdir(inputs_dir):           
            if str(ev_id)+'.in' in str(file):  # augmented ISC phases (plus ad-hoc array phases)
            
                if os.path.isfile(outputs_dir + str(file)[:-2] + 'out'):
                    continue

                # Infile
                print(file) 
                infile = inputs_dir + str(file)

                # Make outfile name
                event = str(file)[:-3]
                outfile = outputs_dir + event + '.out'
                print(event, outfile)

                command = 'echo "update_db=0 isf_stafile=' + station_list_dir+ str(event) +' isf_infile=' + infile + ' isf_outfile=' + str(outfile) + '" | ./iscloc_nodb isf >' + outputs_dir + str(event) + '.log'
                print(command)
                os.system(command)

            if str(ev_id)+'.dat' in str(file):  # original ISC phases
                
                if os.path.isfile(outputs_dir + str(file)[5:-4] + '.ISFout'):
                    continue
                
                # Infile
                print(file) 
                infile = inputs_dir + str(file)

                # Make outfile name
                event = str(file)[5:-4]
                outfile = outputs_dir + event + '.ISFout'
                print(event, outfile)

                command = 'echo "update_db=0 isf_stafile=' + station_list_dir+ str(event) +' isf_infile=' + infile + ' isf_outfile=' + str(outfile) + '" | ./iscloc_nodb isf >' + outputs_dir + str(event) + '.log'
                print(command)
                os.system(command)
        
        
