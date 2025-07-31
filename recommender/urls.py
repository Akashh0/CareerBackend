from django.urls import path
from .views import generate_recommendation_from_input

urlpatterns = [
    path('recommend/', generate_recommendation_from_input, name='generate_recommendation_from_input'),
]
