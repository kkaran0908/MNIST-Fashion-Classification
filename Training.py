import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from skimage import feature
##read the dataset
Xtr=np.loadtxt("TrainData/TrainData.csv")
Ytr=np.loadtxt("TrainLabels.csv")
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
#train the model on the entire dataset
clf = LR(C=.1,n_jobs = -1)
clf.fit(images,Ytr)
##save the model into the system
pickle.dump(clf, open('LogisticReg.pickle', 'wb'))
print("Model Trained Successfully")