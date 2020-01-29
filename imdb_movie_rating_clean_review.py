from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
import sys

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getStemmed_review(review):
    review = review.lower()  #str
    review = review.replace("<br /><br />", " ")
    #tokenize
    tokens = tokenizer.tokenize(review)  #list

    new_tokens = [ token for token in tokens if token not in en_stopwords]  #list

    stemmed_tokens = [ ps.stem(token) for token in new_tokens ]  #list
    cleaned_review = ' '.join(stemmed_tokens)  #str
    return cleaned_review

def getstemmed_document(input_file,output_file):

    output = open(output_file,'w')
    with open(input_file,'r') as f:  
        reviews = f.readlines()

    for review in reviews:
        cleaned_review = getStemmed_review(review)
        print((cleaned_review))
        output.write(cleaned_review)
    output.close


input_file = sys.argv[1]
output_file = sys.argv[2]

getstemmed_document(input_file,output_file)

