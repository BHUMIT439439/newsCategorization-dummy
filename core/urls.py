from django.urls import path
from .import views

urlpatterns = [
    path('', views.PredictCategory, name='PredictCategory'),
]
