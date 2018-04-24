import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import shutil
import os
import os.path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# load the data
data = []
count=0
for root, dirs, files in os.walk("/home/dongsheng/code/CLEF/clef2018-factchecking-master/data/res2"):
    for name in files:
        if name.endswith("txt"):
            with open(root + '/' + name,'r') as f:
                countLine=0
                sents=[]    # first value
                ruling="null"   # second value
                sample=[]
                for line in f:
                    countLine=countLine+1
                    if(countLine==1):
                        ruling=line.strip()
                    else:
                        strs = line.strip().split("\t")
                        if len(strs)<3: # total_Return, rank, year, text
                                continue
                        sents.append(strs[3])
                sample.append(".".join(sents))
                if ruling in ['False','Pants on Fire!','Mostly False']:
                    ruling='False'
                elif ruling in ['Mostly True','','True']:
                    ruling='True'
                elif ruling=='Half-True':
                    ruling='Half-True'
                else:
                    ruling='Unsure'
                sample.append(ruling)
                data.append(sample)
                #print(count,len(sample[0]),sample[1])
                count=count+1
print("data size:",count)               
numpy_array = np.array(data)
X = numpy_array[:,0]
Y = numpy_array[:,1]

# split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# bayes model
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
parameters_nb = {'vect__ngram_range': [(2,2),(2,3),(2,4)],'tfidf__use_idf': (True,False),'clf__alpha': (1e-2, 1e-3),}
gs_clf_nb = GridSearchCV(text_clf, parameters_nb, n_jobs=-1)
gs_clf_nb = gs_clf_nb.fit(X_train, Y_train)
best_score = gs_clf_nb.best_score_
best_paras = gs_clf_nb.best_params_
print("----best para and score for naive bayes---")
print(best_paras)
print(best_score)

print("---- overall for basic naive bayes----")
# fit model and predict
text_clf = text_clf.fit(X_train,Y_train)
predicted = text_clf.predict(X_test)
avg = np.mean(predicted == Y_test)
print(avg)
