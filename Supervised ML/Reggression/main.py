#  2a + 4b + c + 3d + 2e = y

import random
from sklearn.linear_model import LinearRegression

train_x = []
train_y = []
test_x = [[10,20,10,20,10]]

for num in range(1000):
    a = random.randint(0,1000)
    b = random.randint(0,1000)
    c = random.randint(0,1000)
    d = random.randint(0,1000)
    e = random.randint(0,1000)

    train_x.append([a,b,c,d,e])
    train_y.append(2*a + 4*b + c + 3*d + 2*e)

model = LinearRegression()
model.fit(train_x, train_y)

result = model.predict(test_x)
print("Result:",result)
print("coefficients:",model.coef_)
