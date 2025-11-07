"""
URL configuration for c2h5oh_hackaton project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from subjects.views.proces_pickle import ImportDataAPIView
from subjects.views.import_subjects import ImportReadmeDataAPIView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/import-data/<str:subject_code>/', ImportDataAPIView.as_view(), name='import_data'),
    path('api/import-readme/<str:subject_code>/', ImportReadmeDataAPIView.as_view(), name='import_readme'),
]
