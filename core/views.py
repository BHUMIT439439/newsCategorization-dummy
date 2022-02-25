from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from joblib import dump, load
import joblib
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

print("-------------------------------------------------------------")
dataset = pd.read_csv('./NewsCategorizer.csv', sep=",", encoding='utf-8')

dataset['Category'] = dataset['category']
dataset.drop('category',axis='columns', inplace=True)

dataset['Text'] = dataset['headline'] + dataset['short_description']
dataset.drop('headline',axis='columns', inplace=True)
dataset.drop('short_description',axis='columns', inplace=True)

dataset.drop('links',axis='columns', inplace=True)
dataset.drop('keywords',axis='columns', inplace=True)
dataset['CategoryId'] = dataset['Category'].factorize()[0]

def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)
    
dataset['Text'] = dataset['Text'].apply(remove_tags)

def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews
dataset['Text'] = dataset['Text'].apply(special_char)

def convert_lower(text):
    return text.lower()
dataset['Text'] = dataset['Text'].apply(convert_lower)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]

dataset['Text'] = dataset['Text'].apply(remove_stopwords)

def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])

dataset['Text'] = dataset['Text'].apply(lemmatize_word)
# dataset['Text'] = dataset['headline'] + dataset['short_description']
# dataset.head()
print("-------------------------------------------------------------")

@require_http_methods(['GET', 'POST'])
def PredictCategory(request):
    news=""
    if request.method == 'POST':
        cv = CountVectorizer(max_features = 5000)
        cv.fit(dataset.Text)
        text = cv.transform([request.POST['mynews']])
        news=request.POST['mynews']
        model = joblib.load('./oneVsRest.joblib')
        category = model.predict(text)

        if category == [0]:     
            result = "WellNess News"
        elif category == [1]:
            result = "POLITICS News"
        elif category == [2]:
            result = "ENTERTAINMENT News"
        elif category == [3]:
            result = "TRAVEL News"
        elif category == [4]:
            result = "STYLE & BEAUTY News"
        elif category == [5]:
            result = "PARENTING News"
        elif category == [6]:
            result = "FOOD & DRINK News"
        elif category == [7]:
            result = "WORLD News"
        elif category == [8]:
            result = "BUSINESS News"
        elif category == [9]:
            result = "SPORTS News"

        return render(request, 'core/form.html', {'category': result,'news':news})
    return render(request, 'core/form.html', {'category': "", 'news':news})