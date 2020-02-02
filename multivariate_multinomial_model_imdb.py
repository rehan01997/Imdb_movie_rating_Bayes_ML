
x_clean = ['love movi sinc 7 saw open day touch beauti strongli recommend see movi watch famili far mpaa rate pg 13 themat element prolong scene disastor nuditi sexual languag',
'first thing first edison chen fantast believ job cambodian hit man born bred dump gladiatori ring hone craft savag batteri order surviv live mantra kill kill role littl dialogu least line cambodian thai perform compel probabl jet li vehicl danni dog man bred sole purpos fight someon els leash like danni dog much talk bare knuckl fight sequenc choreograph stylist rather design normal brutal fisticuff everyth goe probabl brought sens realism grit see charact slug throat defend live take away other grim gritti dark movi liter figur set apart usual run mill cop thriller product edison play hire gun cambodia becom fugit hong kong run cop pickup gone awri lead chase team led cheung siu fai contend maverick member inspector ti sam lee inclus accept team sin father begin cat mous game dark shade shadow seedier look side hong kong stori work multipl level especi charact studi hit man cop opposit side law see within charact black white shade grey hit man see care side got hook develop feel love girl pei pei bring sens matur tender reveal heart gold cop question tactic attitud make wonder one would buckl will anyth take get job done mani interest moment moral question anti hero despic strategi adopt ask make man make beast tendenc switch side depend circumst dark inner streak us transform man dog dog man dog bite dog grip start never let go end though point mid way seem drag especi tender moment suffer know end pick favourit scene must one market food centr extrem well control deliv suspens edg seat moment listen music score dream hear growl dog highli recommend especi think seen almost everyth cop thriller genr',
'brows discount video bin pick movi 4 88 fifti percent time movi find bin pure crap mean horribl beyond belief half time turn surprisingli good movi much better expect found engag though obvious made amateur direct noth special stori intrigu good thrill expect comedi disappoint thriller movi surprisingli good natur bloodi violenc profan nuditi sex usual movi requir four element pg rate well deserv like sixteen candl f word use twice brief gratuit nude scene wish romanc corey haim love interest could develop film tend plot heavi potenti good subplot push side instead develop chemistri two end watch careless three minut montag romant endeavor end kiss end littl chemistri seem forc dream machin gem good clean entertain quit forgett especi cast unknown except haim also much better expect score 7 10']
y = [1,0,1]

#vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (1,2))
x_vec = cv.fit_transform(x_clean , y_clean).toarray()
print(x_vec.shape)
print(cv.get_feature_names())

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()     #count/frequency
mnb.fit(x_vec,y)

#Mulinomial Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(binarize=0.0)    #presence of words
bnb.fit(x_vec , y ) 

#test vectorize
def getStemmed_review(review):
    review = review.lower()  #str
    review = review.replace("<br /><br />", " ")
    #tokenize
    tokens = tokenizer.tokenize(review)  #list

    new_tokens = [ token for token in tokens if token not in en_stopwords]  #list

    stemmed_tokens = [ ps.stem(token) for token in new_tokens ]  #list
    cleaned_review = ' '.join(stemmed_tokens)  #str
    return cleaned_review


#test
x_test = ['movie was awesome' , ' bad not great ' , ' Great movie !']
xt_clean = [getStemmed_review(review) for review in x_test]
xt_vec = cv.transform(xt_clean).toarray()

y_pred_mnb = mnb.predict(xt_vec)
print("mnb pred:",y_pred_mnb)
print("mnb score",mnb.score(x_vec , y))
print("mnb score prob",mnb.predict_proba(xt_vec))

y_pred_bnb = bnb.predict(xt_vec)
print("Bnb pred:",y_pred_bnb)
print("bnb score",bnb.score(x_vec , y))
print("bnb score prob",bnb.predict_proba(xt_vec))



