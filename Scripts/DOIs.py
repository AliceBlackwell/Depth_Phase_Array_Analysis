import os
import re
import requests
from bs4 import BeautifulSoup
import numpy as np

# Function to fetch network information
def fetch_network_info(code, year,base_url):
    search_url = f"{base_url}{code}"
    #print(search_url)
    
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    #print(soup)
    
    # Find table rows containing the network code
    for row in soup.find_all("a", href=True):
        if row:
            href = row['href']                
            if code in href:
                #print('Matched network code href', href)
                link = "https://www.fdsn.org" + href
                
                response = requests.get(link)
                soup_network = BeautifulSoup(response.text, 'html.parser')
                #print(soup_network.prettify())
                
                # Extract Network Name: it's in the section with <h4> "Network Name"
                tables = soup_network.find_all("table", class_='network-information')
                network_code = network_name = start_year = end_year = doi = "N/A"

                for table in tables:
                    text = table.get_text()

                    # Check for main info table
                    #if "FDSN Code" in text and "Network name" in text:
                    rows = table.find_all("tr")

                    for row in rows:
                        headers = row.find_all("th")
                        values = row.find_all("td")
                        
                        for th, td in zip(headers, values):
                            label = th.get_text(strip=True)
                            value = td.get_text(strip=True)
                            #print(label, value)
                            if label == "FDSN code":
                                network_code = value
                            elif label == "Network name":
                                network_name = value
                                network_name = ' '.join(network_name.split())
                            elif label == "Start year":
                                start_year = value
                            elif label == "End year":
                                end_year = value
                    
                    # Check for DOI table
                    if "Digital Object Identifier (DOI)" in text:
                        doi_tag = table.find("a", href=True)
                        if doi_tag and "doi.org" in doi_tag["href"]:
                            doi = doi_tag["href"]
                #print()
                print('Info:', network_code, 'a', network_name, 'b', start_year, 'c', end_year, 'd', doi)
                
                # Check if the network's operational years match the provided year
                if end_year.isdigit():
                    if start_year <= year <= end_year:
                        return code, network_name, start_year, end_year, doi
                    else:
                        continue
                else:
                    return code, network_name, start_year, end_year, doi
    return code, np.nan, np.nan, np.nan, np.nan

#=========================================================================================

def make_doi_list():
    output_file = "../Processed_DATA/network_years.txt"
    results = set()

    with open(output_file, "w") as f:
        f.write("Name\tYear\n")

    with open("network_info.txt", "w") as file:
            file.write("Network Code".ljust(14) + "Network Name".ljust(120) + "Start Year".ljust(12) + "End Year".ljust(12) + "DOI".ljust(50) + '\n')

    # Walk through top-level directories in the current directory
    for entry in os.listdir("../Processed_DATA"):  # Check Processed_DATA directory for event data directories
        print(entry)
        if os.path.isdir('../Processed_DATA/' + entry):
            print()
            print(entry)
            year = entry[:4]
            data_dir = os.path.join(entry, "Data")
            #print(data_dir)

            for file in os.listdir('../Processed_DATA/' + data_dir): # Check event data directories for MSEED names
                if file.endswith(".MSEED"):
                    parts = file.split(".")
                    if len(parts) >= 3:
                        network = parts[0]
                        results.add((network, year))
                        
    unique_results = list(set(results))        # List and sort unique MSEED network names and years for all event directories in data directory
    unique_results = sorted(unique_results)
    print('DOIs for:', unique_results)
    print('Total no.', len(unique_results))

    # Write results to a text file
    with open(output_file, "a+") as f:
        for network, year in sorted(unique_results):
            f.write(f"{network}\t{year}\n")

    print(f"Saved to {output_file}")
        
    # ======== Search FDSN Website ========
    # List of network codes and years (replace with your actual list)
    #network_codes = [
    #    ("IU", "2000"),
    #    ("US", "2005")]
    network_codes = unique_results

    # Base URL for FDSN Network Codes
    base_url = "https://www.fdsn.org/networks/?search="
     
    # Open a file to write the results
    # Iterate over each network code and year
    for code, year in network_codes:
        print(code, year)
        code, network_name, start_year, end_year, doi = fetch_network_info(code, year, base_url)
        if code:
            with open("../Processed_DATA/network_info.txt", "a+") as file:
                file.seek(0)
                content = file.read()
                print(doi)
                if doi != 'N/A':
                    if doi in content:   # if DOI already in text file, do not write out again.
                        pass #continue
                    else:
                        file.write(str(code).ljust(14) + str(network_name).ljust(120) + str(start_year).ljust(12) + str(end_year).ljust(12) + str(doi).ljust(50) + "\n")
                else:
                    if network_name in content:   # if network_name already in text file, do not write out again.
                        pass #continue
                    else:
                        file.write(str(code).ljust(14) + str(network_name).ljust(120) + str(start_year).ljust(12) + str(end_year).ljust(12) + str(doi).ljust(50) + "\n")

    print("Network information has been written to 'network_info.txt'.")
    print()
    return

    
