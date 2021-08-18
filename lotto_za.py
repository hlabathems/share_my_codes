'''
    This prints out all the lotto results since the first draw.
'''

# Let's import the necessary modules
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools
import requests
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Define a function that randomly draws six numbers from a set of lotto numbers based on given probability. This will be used to predict the upcoming draw.
def draw():

    lotto_numbers = np.arange(1, 52 + 1, 1) # Generate a set of 1 to 52 numbers

    selected_numbers = np.random.choice(lotto_numbers, 6, replace = False) # Randomly select 6 numbers from the set.
    selected_numbers = np.sort(selected_numbers) # Sorted array

    return selected_numbers

# The following function gets all the previous draws.
def previous_draws(url):
    response = requests.get(url)

    # To store previous draws
    drawn = []
    # If the status code is 200, continue
    if response.status_code == requests.codes.ok:
        temporary = [] # Temporary list to store drawn numbers
        # Pass response to BeautifulSoup to deal with the HTML document
        soup = BeautifulSoup(response.text, 'html5lib')
        for li in soup.find_all('li', {'class': 'result medium lotto ball dark ball'}):
            number = int(li.contents[0])
            temporary.append(number)
            if len(temporary) == 6:
                drawn.append(temporary)
                temporary = []
            else:
                continue

    return drawn

# The main function
def main():
    # From year 2000 to now, obtained from https://za.national-lottery.com/
    lotto_years = np.arange(2000, 2022, 1)[::-1]
    all_draws = []
    for year in lotto_years:
        url = 'https://za.national-lottery.com/lotto/results/{}-archive'.format(year)
        all_draws.extend(previous_draws(url))
    
    draws = np.array(all_draws)

    # Print all the results
    print(draws)

    # Calculate the frequency of each number. In other words, how many times has each number been drawn previously?
    flattened_draws = list(itertools.chain(*draws)) # Flatten to convert to 1-D
    df = pd.DataFrame(flattened_draws) # Create pandas DataFrame

    normalized = df.value_counts(normalize = True).sort_index()
    lotto_numbers = normalized.index.tolist()
    probabilities = normalized.values.tolist()

    # Do a histogram to see numbers frequently drawn.
    fig = go.Figure(data = [go.Histogram(x = flattened_draws)])
    fig.show()

    # Predict next draw based on calculated probabilities
    predicted = draw(probabilities)
    print(predicted)

main()

