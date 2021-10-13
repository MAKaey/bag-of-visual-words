from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import cv2


class BagOfVisualWords:

	def __init__(
		self,
		num_clusters: int = 50,
        feature_detector: str = "SIFT"
	) -> None:

		self.num_clusters = num_clusters

		if feature_detector == "SIFT":
			self.detector = cv2.xfeatures2d.SIFT_create()
		elif feature_detector == "ORB":
			self.detector = cv2.ORB_create()
		else:
			print("ValueError: Wrong feature detector name")
			return

		self.kmeans = KMeans(
			n_clusters=num_clusters, 
			init="k-means++", 
			n_init=10,
			max_iter=300
		)

		self.scale = StandardScaler()

		self.best_params = {
			"kernel": "linear",
			"C": 1.0,
			"gamma": 1.0
		}

	def get_descriptors(
		self,
        images: np.ndarray,
    ) -> list:
    
	    descriptors = []
	    for index in range(images.shape[0]):
	        keypoint, descriptor = self.detector.detectAndCompute(images[index], None)
	        descriptors.append(descriptor)

	    return descriptors

	def get_image_features(
		self,
        descriptors: list,
        num_instances: int,
    ) -> np.ndarray:
    
	    features = np.zeros(shape=(num_instances, self.num_clusters))

	    for img_index in range(num_instances):
	        for des_index in range(descriptors[img_index].shape[0]):
	            feature = descriptors[img_index][des_index].reshape(1, -1)
	            index = self.kmeans.predict(feature)
	            features[img_index][index] += 1
	    
	    return features

	def get_optimize_params(
		self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
    
	    params = {
	        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
	        "C": np.linspace(0.1,0.5,10).round(2),
	        "gamma": np.linspace(0.1,0.5,10).round(2)
	    }
	    
	    X = np.dot(features, features.T)
	    y = labels.ravel()
	    
	    grid_search = GridSearchCV(SVC(), params)
	    grid_search.fit(X, y)

	    return grid_search.best_params_

	def train(
		self,
        images: np.ndarray,
        labels: np.ndarray,
        optimize_params: bool = True
    ) -> None:
    
	    num_instances = images.shape[0]
	   	
	    descriptors = self.get_descriptors(images)
	    self.kmeans.fit(np.vstack(descriptors))

	    features = self.get_image_features(descriptors, num_instances)
	    features = self.scale.fit_transform(features)
	    self.code_book = features

	    if optimize_params:
		    self.best_params = self.get_optimize_params(features, labels)

	    self.svm = SVC(
	        kernel = self.best_params["kernel"],
	        C = self.best_params["C"],
	        gamma =  self.best_params["gamma"]
	    )
	    self.svm.fit(features, labels.ravel())

	def test(
		self,
        images: np.ndarray, 
        labels: np.ndarray,
    ) -> np.ndarray:

	    num_instances = images.shape[0]
	    
	    descriptors = self.get_descriptors(images)
	    features = self.get_image_features(descriptors, num_instances)
	    features = self.scale.transform(features)
	    predictions = self.svm.predict(features)
	    
	    correct = (predictions == labels.ravel()).sum()
	    accuracy = accuracy_score(labels, predictions) * 100
	    
	    print(f"        Total instances: {num_instances}")
	    print(f"Correct classifications: {correct}")
	    print(f"         Accuracy score: {round(accuracy, 3)}")
	    
	    return predictions