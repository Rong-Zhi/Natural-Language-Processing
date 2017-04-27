import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

# print(digits.data)
# print(digits.target)  #   digits.target is the actual label
                        #   we've assigned to the digits data.
# print(digits.images[0])

#Normalise the data into -1 -- 1
clf = svm.SVC(gamma=0.001, C=100)

# print(len(digits.data))

x,y = digits.data[:-5],digits.target[:-5]

clf.fit(x,y)

print('Prediction:',clf.predict(digits.data[-5]))

plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show() 