import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import joblib

df = pd.read_csv(r'C:\Users\meysam-sadat\Downloads\Compressed\emails.csv')

my_vectorize = CountVectorizer()

x = my_vectorize.fit_transform(df['text'])
y = df.spam.values



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
truncated  = TruncatedSVD(n_components=1000)
x_train = truncated.fit_transform(x_train)
x_test = truncated.transform(x_test)

clf = MLPClassifier(hidden_layer_sizes=(200,50))
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

joblib.dump(clf,'spam_pam_detector.joblib')
model = joblib.load('spam_pam_detector.joblib')
prediction = model.predict(x_test)

