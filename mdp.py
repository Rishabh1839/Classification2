# Using supervised learning and labeled training data
# classification of minivans and sports cars
from sklearn import tree

print("Welcome to Machine Learning Classification created by Rishabh Singh")
print("What is This program about?")
print("-Data Classification is the problem of identifying to which")
print("of a set of categories a new observation belongs on the basis of ")
print("a training set of data containing observations whose category membership is known.")
print("=" * 70)

print("A little Overview of what our program does here.")
print("We have a set of resolutions and the mega pixels that each resolution provides.")
print("This will be our choice of classification for this program.")
print("We set out features labeled as resolutions and Mega pixels")
print("For example a Feature would be equal to [resolution] x [mega pixels]")
print("Our labels are set to 1's and 0's where 0 is classified as High Definition")
print("And 1's stand for Ultra High Definition resolution")
print("We implement a decision tree algorithm amd train the program so the fit is finding its patterns in the data.")
print("We later place unknown data.")

# include features of the image by resolution x mega pixels
# using the following resolutions in sequence
# 1280 x 720, 1920 x 1080, 3840 x 2160 and 7680 x 4320
# the mega pixels are multiplying the following resolutions above in the list
# to be able to find it's mega pixels
features = [[720, 0.9], [1080, 2.0], [3840, 8.2], [7680, 33.1]]
# labeled the training data
# labels = ["High Def", "High Def", "Ultra High Def", "Ultra High Def"]
labels = [0, 0, 1, 1]

# create a classifier which is a decision tree
clf = tree.DecisionTreeClassifier()

# Do the actual training here for the ML
# Fit is finding the patterns in data
clf = clf.fit(features, labels)

# place unknown data
# predicting its a High Def with 1280 x 720 resolution with 0.9 mega pixels or higher
print("Here we predicted a label of 0 since we set an output for predicting")
print("its a High Def with 1280 x 720 resolution with 0.9 mega pixels or higher")
print(clf.predict([[780, 1.3]]))
# predicting its a Ultra High Def with 7680 x 4320 resolution with 33.1 mega pixels or higher
print("Here we predicted a label of 1 since we set an output for predicting")
print("its a Ultra High Def with 7680 x 4320 resolution with 33.1 mega pixels or higher")
print(clf.predict([[7820, 40]]))
# predicting its a High Def with 1920 x 1080 resolution with 2.0 mega pixels or higher
print("Here we predicted a label of 0 since we set an output for predicting")
print("its a High Def with 1920 x 1080 resolution with 2.0 mega pixels or higher")
print(clf.predict([[2049, 4.0]]))
# predicting its a High Def with 3840 x 2160 resolution with 2.0 mega pixels or higher
print("Here we predicted a label of 1 since we set an output for predicting")
print("its a High Def with 3840 x 2160 resolution with 2.0 mega pixels or higher")
print(clf.predict([[3900, 6.0]]))