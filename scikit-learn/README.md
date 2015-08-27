### Scikit-learn

#### Preprocess

```python
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
```

##### Load Data
```python
boston = datasets.load_boston()
print boston.DESCR
X, y = boston.data, boston.target
```

##### Create Data
```python
reg_data = datasets.make_regression(100, 10, 5, 2, 1.0)
reg_data[0].shape
classification_set = datasets.make_classification(weights=[0.1])
np.bincount(classification_set[1])
blobs = datasets.make_blobs()
```

##### Scaling Data
```python
X_2 = preprocessing.scale(X)
X_2.mean(axis=0)
X_2.std(axis=0)

my_scaler = preprocessing.StandardScaler()
my_scaler.fit(X)
my_scaler.transform(X)

my_minmax_scaler = preprocessing.MinMaxScaler()
my_minmax_scaler.fit(X)
my_minmax_scaler.transform(X)

normalized_x = preprocessing.normalize(X)
```

##### Binarization
```python
new_target = preprocessing.binarize(target, threshold=target.mean())

bin = preprocessing.Binarizer(target.mean())
new_target = bin.fit_transform(target)

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
new_target = label_binarizer.fit_transform(target)
```

##### Categorize
```python
text_encoder = preprocessing.OneHotEncoder()
text_encoder.fit_transform(d).toarray()

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
dv.fit_transform(dict).toarray()
```

##### Missing Value
```python
impute = preprocessing.Imputer(strategy='mean', missing_values=-1)
x_prime = impute.fit_transform(x)

df = pd.DataFrame(X, columns=feature_names)
df.fillna(df.mean())
```

##### PCA
```python
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca = decomposition.PCA(n_components=.98)
reduced_data = pca.fit_transform(data)
print pca.explained_variance_ratio_
```

##### Factor Analysis
```python
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)
two_dim = fa.fit_transorm(data)
```

##### Pipleline
```python
from sklearn import pipeline
from sklearn import decomposition
impute = preprocessing.Imputer()
scaler = preprocessing.StandardScaler()
pca = decomposition.PCA()
pipe = pipeline.Pipeline([('impute',impute),('scaler',scaler), ('pca',pca)])
new_mat = pipe.fit_transform(mat)
mat = pipe.inverse_transform(new_mat)
```

###### Gaussian Process
```python
from sklearn.gaussian_process import GaussianProcess
gp = GaussianProcess()
gp.fit(X, y)
preds = gp.predict(X)
```

#### Linear Model

##### MakeRegression
```python
reg_data, reg_target = make_regression(n_sample=200, n_features=500, n_informative=10, noise=2)
```

##### LinearRegression
```python
# fit and predict
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(data, target)
predictions = lr.predict(data)
print lr.coef_
# evaluate
plt.hist(target - predictions, bins = 50)
np.mean(target - predictions) # should be very close to 0
# q-q plot
from scipy.stats import probplot
probplot(target - predictions)
# MSE & MAD
def MSE(target, predictions): np.mean(np.power(target - predictions, 2))
def MAD(target, predictions): np.mean(np.abs(target - predictions)) 
print MSE(target, predictions)
print MAD(target, predictions)
# coefficient distribution
def bootstrap(lr, data, target, idx, n_bootstrap = 1000):
	len_data = len(target)
	subsample_size = np.int(0.5 * len_data)
	subsample = lambda : np.random.choice(np.arange(0, len_data), size=subsample_size)
	coefs = np.ones(n_bootstrap)
	for i in range(n_bootstrap):
		subsample_idx = subsample()
		subsample_X = data[subsample_idx]
		subsample_y = target[subsample_idx]
		lr.fit(subsample_X, subsample_y)
		coefs[i] = lr.coef_[0]
	return coefs

plt.hist(coefs, bins=50)
np.percentile(coefs, [2.5,97.5])
```

##### Ridge Regression
```python
r = Ridge()
r.fit(data, target)
predictions = lr.predirct(data)

# RidgeCV
from sklearn.linear_model import RidgeCV
rcv = RidgeCV(alphas=np.array([.1,.2,.3,.4]))
rcv.fit(data, target)
print rcv.alpha_
```

##### Lasso
```python
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(data, target)
print np.sum(lasso.coef_ != 0)
lassocv = LassoCV()
lassocv.fit(data, target)
print np.sum(lassocv.coef_ != 0)
```

##### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)
prediction = lr.predict(X)
print (prediction == label).sum().astype(float) / label.shape[0]
```

#### Cluster

##### MakeBlob
```python
blobs, classes = make_blobs(500, centers=3)
```

##### KMeans
```python
from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=3)
kmean.fit(blobs)
print kmean.cluster_centers_
kmean.transform(blobs) # calculate the distance between point and center
# metric:silhouette
from sklearn import metrics
silhouette_samples = metrics.silhouette_samples(blobs, kmean.labels_)
np.column_stack((classes[:5], silhouette_samples[:5]))
print silhouette_samples.mean()
# metric:mutual_info
from sklearn import metrics
metrics.normalized_mutual_info_score(ground_truth,kmeans.labels_)
# metric:inertia
print kmean.inertia_
# minibatch
from sklearn.cluster import MiniBatchKMeans
minibatch - MiniBatchKMeans(n_clusters=3)
kmeans = KMeans(n_clusters=3)
print np.diag(pairwise.pairwise_distance(kmeans.cluster_centers_, minibatch.cluster_centers_))
# find outlier
kmeans.fit(X)
distances = kmeans.transform(X)
sorted_idx = np.argsort(distances.ravel())[::-1]
new_X = np.delete(X, sorted_idx, axis=0) # remove outlier
```

##### Find Closest Point
```python
# real point
points. labels = make_blobs()
distances = pairwise.pairwise_distances(points, metric='l2')
ranks = np.argsort(distances[0])
points[ranks]

# bool hamming
X = np.random.binomial(1, .5, size=(2,4)).astype(np.bool)
pairwise.pairwise_distances(X, metric='hamming')
```

##### GMM
```python
from sklearn.mixture import GMM
gmm = GMM(n_components=2)
X = np.row_stack((class_A, class_B))
y = np.hstack((np.ones(100), np.zeros(100)))

train = np.random.choice([True, False], 200)
gmm.fit(X[train])
gmm.predict(~X[train])
```

##### KNN Regression
```python
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors=10)
knnr.fit(X,y)
print np.power(y - knnr.predict(X), 2).mean() # MSE
```

#### Classifier

##### Make Classification
```python
from sklearn import datasets
X, y = datasets.make_classification(n_samples=1000, n_features=3, n_redundant=0, n_informative=3, weights=[.2,.8])
```
##### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=10)
dt.fit(X, y)
preds = dt.predict(X)
# get feature importance
dt_ci = DecisionTreeClassifier(compute_importances=True)
dt.fit(X, y)
ne0 = dt.feature_importances_ != 0
y_comp = dt.feature_importances_[ne0]
```

##### Random forests
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
```

##### SVM
```python
from sklearn.svm import SVC
base_svm = SVC()
base_svm.fit(X, y)
from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(X, y)
```

##### MultiClass Classification
```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
mlr = OneVsRestClassifier(LogisticRegression(), n_jobs=2)
mlr.fit(X, y)
mlr.predict(X)
```

##### LDA
```python
from sklearn.lda import LDA
lda = LDA()
lda.fit(X.ix[:,:-1], X.ix[:,-1])
from sklearn.metrics import classification_report
print classification_report(X.ix[:, -1].values, lda.predict(X.ix[:,-1]))
# QDA
from sklearn.pda import QDA
qda = QDA()
```

##### SGD
```python
from sklearn import linear_model
bgd_clf = linear_moel.SGDClassifier()
```

##### Naiive Bayes
```python
from sklearn.datasets import fetch_20newsgroups
categories = ["rec.autos", "rec.motorcycles"]
newgroups = fetch_20newsgroups(categories=categories)
print "\n".join(newgroups.data[:1])
print newgroups.target_names[newgroups.target[:1]]

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
bow = count_vec.fit_transform(newgroups.data)
bow = np.array(bow.todense())

from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
mask = np.random.choice([True, False], len(bow))
clf.fit(bow[mask], newgroups.target[mask])
predictions = clf.predict(bow[~mask])

# multi nb
multinb = naive_bayes.MultinomialNB()
```

#### PostProcess

##### K-fold CV
```python
from sklearn.cross_validation import KFold
kfold = KFold(len(y), n_folds=4)
for i, (train, test) in enumerate(kfold):
```


