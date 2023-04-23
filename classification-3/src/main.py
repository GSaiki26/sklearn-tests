# Libs
import matplotlib.pyplot as plt
from numpy import ones
from pandas import read_csv
from seaborn import relplot, scatterplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Get the csv
uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
csv = read_csv(uri)

# Rename and change the unfinished column
csv = csv.rename(columns={
    'unfinished': 'finished'
})
csv['finished'] = csv['finished'].map({
    0: 1,
    1: 0
})

# Get the x and y
SEED = 1000

x = csv[['expected_hours', 'price']]
y = csv['finished']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=SEED, stratify=y)
print(f'Train: {len(x_train):>6} | Test: {len(x_test):>6}')

# Get the plot
scatterplot(csv, x='expected_hours', y='price', hue='finished')
plt.savefig('data/scatterplot.png')

relplot(csv, x='expected_hours', y='price', hue='finished', col='finished')
plt.savefig('data/relplot.png')

# Train the model
model = LinearSVC()
model.fit(x_train, y_train)

# Predict
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict) * 100
print(f'The accuracy from the model is: {round(accuracy, 2)}%')

# Check the baseline
baseline = ones(len(x_test))
baseline_accuracy = accuracy_score(y_test, baseline) * 100
print(f'The accuracy from the baseline is: {round(baseline_accuracy, 2)}%')
