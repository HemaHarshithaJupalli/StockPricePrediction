from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage

from plotly.offline import plot
import plotly.graph_objects as go
import yfinance as yf

import csv

from stock_app.lstm_prediction import*

# Define a global variable to store valid tickers
valid_tickers = []

def mail(req):
	if req.method=="POST":
		sndr=req.POST.get('sn','')
		sbj=req.POST.get('sub','')
		m=req.POST.get('msg','')
		if sndr and sbj and m:
			email=EmailMessage(sbj,m,to=['kavyasri_chirumamilla@srmap.edu.in'])
			email.reply_to=[sndr]
			email.send()
			messages.success(req,"Mail sent Successfully")
			return redirect('/mal')
		else:
			messages.error(req,"Mail not Sent")
			return redirect('/mal')
	return render(req,'ht/mail.html')

def no_input(req):
     messages.error(req,"Enter a stock ticker")
     return render(req,'ht/index.html')

def load_valid_tickers():
    global valid_tickers
    path = 'stock_app/Data/Tickers.csv'  # Update with the actual file path
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        next(reader)
        for row in reader:
            valid_tickers.append(row['Symbol'])

# Call load_valid_tickers function to load tickers when the module is imported
load_valid_tickers()
# Create your views here.
def index(req):
    user = req.user if req.user.is_authenticated else None
    return render(req,'ht/index.html',{'user': user})

def predict(req,company):
    company=company.upper()
    if company not in valid_tickers:
        messages.error(req,"Invalid Ticker")
        return redirect('/')
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = company.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        messages.error(req,"There is been a issue..Please try again after some time")

    if df.empty:
        messages.error(req, "No data available for the specified ticker.")
        return redirect('/')
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward",),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')

    predicted_result_df = lstm_prediction(company)

    return render(req,'ht/predict.html',context= {'plot_div':plot_div,
                                                 'predicted_result_df': predicted_result_df,
                                                 })


def login_user(req):
    if req.method=="POST":
        username=req.POST.get('username')
        password=req.POST.get('password')
        user = authenticate(username=username,password=password)
        if user is None:
            messages.error(req,"Invalid username or password")
            return redirect('/log/')
        else:
            login(req,user)
            messages.success(req,("Logged in Sucessfully"))
            return redirect('/')
    return render(req,'ht/login.html')


def logout_user(req):
    logout(req)
    messages.success(req,"You have been logged out.")
    return render(req,'ht/index.html')


def register(req):
    if req.method=='POST':
        username = req.POST.get('username')
        password = req.POST.get('password')

        user=User.objects.filter(username=username)
        if user.exists():
            messages.info(req,"Username already exists")
            return redirect('/reg/')
        
        user = User.objects.create_user(
            username=username
        )

        user.set_password(password)
        user.save()
        messages.info(req,"Account Created Successfully!")
        return redirect('/log/')
    return render(req,'ht/signup.html')