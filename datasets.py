from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os


class ObjectDataset:
    
    object_categories = {
        0: "Soccer",
        1: "Accordion",
        2: "Motorbike",
        3: "Dollar"
    }
    
    def __init__(
        self,
        root_path: str,
        mode: str
    ) -> None:
        
        dataset_path = os.path.join(root_path, mode)
        self.files = []

        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                self.files.append(file_path)
        np.random.shuffle(self.files)
    
    def __getitem__(self, index: int) -> [np.ndarray, int]:
        
        image_path = self.files[index]
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (150, 150))
        label = self.get_label(image_path)
        
        return image, label
    
    def __len__(self) -> int:
        
        return len(self.files)
    
    def get_label(self, path: str) -> int:
        
        category = path.split("/")[3]
        key = list(self.object_categories.values()).index(category)
        
        return key

class FlowerDataset:
    
    flower_categories = {
        0: "Dandelion",
        1: "Sunflowers",
        2: "Daisy",
        3: "Roses",
        4: "Tulips"
    }
    
    def __init__(
        self,
        root_path: str,
        mode: str
    ) -> None:
        
        self.mode = mode
        self.train_files = []
        self.test_files = []
        
        dataset_path = root_path
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            files = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                files.append(file_path)
            
            train_file, test_file = train_test_split(files, test_size=0.15, train_size=0.85, shuffle=True)
            self.train_files.extend(train_file)
            self.test_files.extend(test_file)
        
    def __getitem__(self, index: int) -> [np.ndarray, int]:
        
        if self.mode == "Train":
            image_path = self.train_files[index]
        elif self.mode == "Test":
            image_path = self.test_files[index]
        else:
            print("ValueError: Wrong mode value.")
            return
        
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (150, 150))
        label = self.get_label(image_path)
        
        return image, label
    
    def __len__(self) -> int:
        
        if self.mode == "Train":
            return len(self.train_files)
        elif self.mode == "Test":
            return len(self.test_files)
        else:
            print("ValueError: Wrong mode value.")
            return
    
    def get_label(self, path: str) -> int:
        
        category = path.split("/")[2]
        key = list(self.flower_categories.values()).index(category)
        
        return key

def load_dataset(
        dataset: [ObjectDataset, FlowerDataset],
        root_path: str,
        mode: str
    ) -> [np.ndarray, np.ndarray]:
    
    dataset = dataset(root_path, mode)
    images = []
    labels = []
    
    for index in range(len(dataset)):
        image = dataset[index][0]
        label = dataset[index][1]

        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels).reshape(-1,1)
    
    return images, labels