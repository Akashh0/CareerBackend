from django.urls import path
from .views import generate_recommendation

urlpatterns = [
    path('recommend/', generate_recommendation, name='generate_recommendation'),
]
