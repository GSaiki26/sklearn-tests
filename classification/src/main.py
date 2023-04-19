# Libs
from sklearn.svm import LinearSVC

# Data
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
test_kangaroo = [0, 1, 1]
test_human = [1, 1, 0]

print(model.predict([test_kangaroo, test_human]))