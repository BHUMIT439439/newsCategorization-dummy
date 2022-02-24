from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import joblib


@require_http_methods(['GET', 'POST'])
def PredictCategory(request):
    print("-------------------------------------------------------------")
    dataset = pd.read_csv('./NewsCategorizer.csv', sep=",", encoding='utf-8')
    dataset['Text'] = dataset['headline'] + dataset['short_description']
    dataset.head()
    print("-------------------------------------------------------------")
    if request.method == 'POST':
        cv = CountVectorizer(max_features = 5000)
        cv.fit(dataset.Text)
        text = cv.transform([request.POST['text']])
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

        return render(request, 'core/form.html', {'category': result})
    return render(request, 'core/form.html', {'category': ""})