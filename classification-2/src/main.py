# Libs
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
csv = pd.read_csv(uri)


x = list(csv[['home', 'how_it_works', 'contact']].values)
y = list(csv['bought'].values)

half = int((len(x)/2)-1)

x_test = x[half:int(len(x)-1)]
x = x[0:half]

y_test = y[half:int(len(y)-1)]
y = y[0:half]

model = LinearSVC()
model.fit(x, y)

predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict) * 100
print('%.2f%%' % accuracy)
