# Libs
from numpy import ndarray
from sklearn.svm import LinearSVC

# Data
SPACE = 10

# Human 0 | Kangaroo 1
# It has 4 members?
# It's tall?
# It has a pocket?
kangaroos = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
]

humans = [
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
]

x = humans + kangaroos
y = [0, 0, 0, 1, 1, 1]

# Code
model = LinearSVC()
model.fit(x, y)

# Test
tests = [
    [0, 1, 1],
    [1, 1, 0]
]
results = [
    1,
    0
]

predict = model.predict(tests)
corrects: ndarray = (results == predict)  # type: ignore
print(f'Results: Accuracy {corrects.sum()}/{len(tests)}')
print(f'The model said:')

for index, test in enumerate(tests):
    print(f'Test: {str(test):<4} | Predicted: {str(predict[index]):<4}')
