from django.urls import path,include
from  stock_app import views

urlpatterns=[
    path('',views.index,name='ind'),
    path('log/',views.login_user,name="log"),
    path('reg/',views.register,name='reg'),
     path('mal/',views.mail,name="ml"),
    path('logout/',views.logout_user,name='logout'),
    path('predict/<str:company>/',views.predict,name="predict"),
    path('predict/',views.no_input,name="predict"),
]
