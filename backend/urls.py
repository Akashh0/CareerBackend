from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('recommender.urls')),
    path('', lambda request: HttpResponse("🚀 Career Recommendation API is running!")),
]
