# Libs
from numpy import ndarray
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
csv = pd.read_csv(uri)


x = csv[['home', 'how_it_works', 'contact']].values
y = csv['bought'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=0, stratify=y)  # type: ignore

print(len(x_train), len(x_test))
print(len(y_train), len(y_test))

model = LinearSVC()
model.fit(x_train, y_train)

predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict) * 100
print('%.2f%%' % accuracy)
