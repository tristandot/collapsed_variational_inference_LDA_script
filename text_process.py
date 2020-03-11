import numpy as np
import pandas as pd
import nltk


#Function to transform tokenized string texts into list of words indices
def text_to_indices(text):

    if len(text) > 1:
        flat_text = [item for sublist in text for item in sublist]
    else:
        flat_text = text

    fdist = nltk.FreqDist(flat_text)

    df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
    df_fdist.columns = ['Frequency']

    df_fdist.sort_values(by=['Frequency'], ascending=False, inplace=True)

    number_of_words = df_fdist.shape[0]
    df_fdist['word_index'] = list(np.arange(number_of_words)+1)

    frequency = df_fdist['Frequency'].values
    word_index = df_fdist['word_index'].values
    df_fdist['word_index'] =  word_index-1

    word_dict = df_fdist['word_index'].to_dict()

    text_numbers = []
    for string in text:
        string_numbers = [word_dict[word] for word in string]
        text_numbers.append(string_numbers)

    return (text_numbers,word_dict)
