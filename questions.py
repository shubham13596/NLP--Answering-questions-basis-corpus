import nltk
import sys
import os
import string
import math

from nltk import word_tokenize
from nltk.corpus import stopwords

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

## query density is yet to be included!

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_strings = {} #dictionary mapping file name to its content within corpus directory
    all_files = [] # list of all files within corpus

    path1 = os.path.join(os.getcwd(), 'corpus') # path1 is the path to the corpus directory
    all_files = os.listdir(path1) # this lists all the files within corpus directory 
    
    for x in range (len(all_files)):
        path2 = os.path.join(path1, all_files[x])
        with open (path2, 'r', encoding= "utf8") as f:
            lines = f.read()
            file_strings[all_files[x]] = lines

    return file_strings


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words_in_document = []

    for word in word_tokenize(document):
        if word not in string.punctuation and word not in stopwords.words('english'):
            words_in_document.append(word.lower())


    return words_in_document


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    all_words = set() # set of all words
    frequency = [] # list for count of occurence of a word across documents

    for document in documents:
        for x in range(len(documents[document])):
            all_words.add(documents[document][x])
        

    for x in range(len(all_words)):
        frequency.append(0)
    
    idf_values = {} # new dictionary
    idf_values = dict(zip(all_words, frequency)) 


    for word in all_words:
        for document in documents:
            if word in documents[document]:
                idf_values[word] = idf_values[word] + 1

    for x in idf_values:
        idf_values[x] = math.log(len(documents)/idf_values[x]) #calculating idf value

    return idf_values


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_tf_idf = dict.fromkeys(files.keys(),0)

    for word in query:
        for file in files:
            tf_idf = (files[file].count(word)) * idfs[word]
            file_tf_idf[file] = file_tf_idf[file] + tf_idf
    

    filenames = sorted(file_tf_idf, key = file_tf_idf.get, reverse = True)

    return filenames[:n]



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    #defining a dictionary which maps each sentence to its score and query term density
    sentences_idf = {} 

    for sentence, sentence_words in sentences.items():
        score = 0
        for word in query:
            if word in sentence_words:
                score = score + idfs[word]

        if score != 0: 
            density = sum([sentence_words.count(x) for x in query])/len(sentence_words)
            #assigning idf score and density as a tuple to each sentence(key)
            sentences_idf[sentence] = (score, density) 
    
    
    #Defining a function which will be used as the key for the sorted() function parameter. Returns the tuple value
    def sort_key(item):
        return (item[1][0], item[1][1])
    
    sorted_tuples = sorted(sentences_idf.items(), key = sort_key, reverse = True)

    sorted_scores = [a for a, b in sorted_tuples]

    return sorted_scores [:n]
    

if __name__ == "__main__":
    main()
