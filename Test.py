import warnings
warnings.filterwarnings("ignore")
##import all the required libraries
import pickle
import pandas as pd
#reading the test data
Xtr = np.loadtxt('TestData.csv')
#reshaping the image into 28*28 size
images = []
for img in Xtr:
    images.append(img.reshape(28,28))
#extracting the edge features from the images
im_edge = []
for im in images:
    im_edge.append(feature.canny(im, sigma=5))

#flatten the entire dataset
images = []
for im in im_edge:
    images.append(im.flatten())
images = np.array(images)

#load the pretrained model
model = pickle.load(open('LogisticReg.pickle', 'rb'))
prediction = model.predict(images)
np.savetxt("myPredictions.csv", prediction) 