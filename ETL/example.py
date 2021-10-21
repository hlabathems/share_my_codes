# Short E(xtract), T(ransform), L(oad) example

import pandas as pd
from bs4 import BeautifulSoup
from astroquery.sdss import SDSS
import numpy as np
import requests
import sys
import warnings

warnings.filterwarnings('ignore')

class ETL():
    def __init__(self, username, password, source_url):
        self.source_url = source_url  # The URL of the AGN key website
        self.username = username
        self.password = password

    def extract(self):
        # The actual work is done here
        with requests.Session() as s:
            # Login to AGN Key website
            response = s.post(self.source_url, auth = (self.username, self.password))

            # If the status code is 200, continue
            if response.status_code == requests.codes.ok:
                # Pass response to BeautifulSoup to deal with the HTML document
                soup = BeautifulSoup(response.text, 'html5lib')
        
                # Empty lists to store data, these are the column names from the website in order
                agn_names, ra, dec, agn_type, z, n_nights, last_obs = [], [], [], [], [], [], []
        
                # Now use the magic of BeautifulSoup to handle the HTML, skip the header and the last two rows as they are not important
                for tr in soup.find_all('tr')[1:-3]:
                    td = tr.find_all('td')

                    agn_names.append(td[0].text)
                    ra.append(td[1].text)
                    dec.append(td[2].text)
                    agn_type.append(td[3].text)
                    z.append(td[4].text)
                    n_nights.append(td[5].text)
                    last_obs.append(td[6].text)

                # Convert to arrays
                agn_names = np.array(agn_names)
                ra = np.array(ra)
                dec = np.array(dec)
                agn_type = np.array(agn_type)
                z = np.array(z)
                n_nights = np.array(n_nights)
                last_obs = np.array(last_obs)
            
                # Create dataframe so that the data is nicely formatted and easy to read
                data = {
                        'AGN': agn_names,
                        'RA': ra,
                        'DEC': dec,
                        'Type': agn_type,
                        'z': z,
                        'N_nights': n_nights,
                        'Last_Obs': last_obs
                }

                self.df = pd.DataFrame(data)

    def transform(self):
        # Change data type of columns
        # If 'coerce', then invalid parsing will be set as NaN
        self.df[['RA','DEC', 'z', 'N_nights']] = self.df[['RA','DEC', 'z', 'N_nights']].apply(pd.to_numeric, errors = 'coerce')
        self.df[['AGN','Type', 'Last_Obs']] = self.df[['AGN','Type', 'Last_Obs']].astype(str)
        
        self.df = self.df[self.df['DEC'].notna()] # Take the rows where DEC is not NAN
        self.df = self.df[self.df['z'].notna()] # Take the rows where z is not NAN
    
        # To observe with SALT, the declination must lie between -76 and +11.25
        dec_start, dec_end = -76.0000, 11.25
        mask = (self.df['DEC'] > dec_start) & (self.df['DEC'] < dec_end)
        visible_with_salt = self.df[mask]
        
        self.visible_with_salt = visible_with_salt
    
    def load(self):
        # Save as CSV file
        self.visible_with_salt.to_csv()

run = ETL('add_username', 'add_password', 'http://dark.physics.ucdavis.edu/~hal/cgi-bin/agnagent/agnkeymain.cgi')

run.extract()
run.transform()
run.load()
