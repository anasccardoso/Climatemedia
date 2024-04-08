import pandas as pd
from pandas import Series
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Import dataset
df = pd.read_excel('tweets.xlsx') # add the excel file containing the tweets for the three languages (portuguese, spanish and english)
print(df)

# Extract date and time
df['Created_at'] = pd.to_datetime(df.Created_at, format='%Y-%m-%d %H:%M:%S', utc = True)
df['Date'] = pd.to_datetime(df['Created_at'], utc=True).dt.date
df['Time'] = pd.to_datetime(df['Created_at'], utc=True).dt.time
print(df)

# Extract year
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m', utc = True)
df["Year"] = df["Date"].dt.year
print(df)

# Frequency of tweets per year
def f(x): # function to calculate the daily frequency of tweets
     return Series(dict(Number_of_tweets = x['Text'].count(),
                        ))

daily_count = df.groupby(df.Date).apply(f)
print(len(daily_count))
daily_count.head(5)

daily_count['index'] = daily_count.index
daily_count = daily_count.rename(columns = {'index':'Date'})
daily_count = daily_count.reset_index(drop=True)
daily_count = daily_count[['Date', 'Number_of_tweets']]
#daily_count['Date'] = daily_count['Date'].dt.date.astype(str)
daily_count.tail(5)
print(daily_count)

# Convert the column (it's a string) to datetime type
datetime_series = pd.to_datetime(daily_count['Date'])
# Create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)
# Datetime_index
period_index = pd.PeriodIndex(datetime_index, freq='M') # get the date at month level
# Period_index
daily_count = daily_count.set_index(period_index)
# Drop the Date column
daily_count.drop('Date',axis=1,inplace=True)
daily_count = daily_count.reset_index()
print(daily_count)

# Count the frequency of tweets by month and year
count_series = daily_count.groupby(["index"]).Number_of_tweets.sum().reset_index()
count_series = count_series.set_index(count_series["index"])
count_series.drop('index',axis=1,inplace=True)
print(count_series)

# Anomaly detection to verify the significance of peaks
# Import the Isolation Forest algorithm
from sklearn.ensemble import IsolationForest
# Initialize Isolation Forest
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), max_features=1.0)
# Train
clf.fit(count_series)
# Find anomalies
count_series['Anomaly'] = clf.predict(count_series)
# Save anomalies to a separate dataset for visualization purposes
anomalies = count_series.query('Anomaly == -1')
print(anomalies)

# Plot the frequency of tweets by month and year
b1 = go.Scatter(x=count_series.index.astype(str),
                y=count_series['Number_of_tweets'],
                name="Normal occurences",
                mode='lines+markers',
                marker=dict(color='#413839', size=20,
                            line=dict(color='#413839', width=1)),
                line= dict(width=5)
               )
b2 = go.Scatter(x=anomalies.index.astype(str),
                y=anomalies['Number_of_tweets'],
                name="Detected peaks",
                mode='markers',
                marker=dict(color='#CF1020', size=30, symbol='star',
                            line=dict(color='#CF1020', width=1)),
                line= dict(width=5)
               )

data = [b1, b2]
fig = go.Figure(data=data)
fig.update_xaxes(tickfont_size=32, tickfont_color='black', tickvals=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], ticktext=["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"])
fig.update_yaxes(tickfont_size=32, tickfont_color='black')
fig.update_layout(font_family="arial", legend_font_size=32, legend_font_color="black", xaxis_title="Year", yaxis_title="Number of posts", font=dict(size=32))
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'

})

fig.show()

fig.write_html("frequency_plot.html")
