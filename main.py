from time import time
from random import choice
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from urllib import request
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from PIL import Image
from numpy import array

class ImageData:
    """Class to scrape google images, take the numerical data of images based on a label and store them in a dictionary
    """
    
    def __init__(self, labels):
        self.linkBoilerplate = f"https://www.google.com/search?q={0}%20emotion&tbm=isch&hl=en&tbs=isz:i&sa=X&ved=0CAQQpwVqFwoTCLCItIuesvkCFQAAAAAdAAAAABAC&biw=1522&bih=746"
        self.labels = {label: [] for label in labels}
        self.getImageData()
        
    @staticmethod
    def openURL(url):
        req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}) # to avoid security concerns
        return request.urlopen(req).read()
    
    def getImageData(self):
        print("Gathering image data ...")
        for label in self.labels.keys():
            imageLink = self.linkBoilerplate.format(label)
            htmlData = self.openURL(imageLink)
            soupData = BeautifulSoup(htmlData, 'html.parser').find_all('img')[1:]
            for item in soupData:
                img = request.urlretrieve(item['src'], '_.png')
                img = array(Image.open('_.png'))
                self.labels[label].append(img)
            print(f"Found {len(self.labels[label])} images for label '{label}'")
        print()
    
    def getImage(self, label):
        return choice(self.labels[label])
        

def main(df):
    start = time()

    print(f"Dataset size: {df.shape}\n")
    X = df.iloc[:, 2]
    y = df.iloc[:, 1]
    imageData = ImageData(y.unique())

    emotionEncode = LabelEncoder().fit(y)
    y = emotionEncode.transform(y)

    print(f"X shape after count vectorzing: {X.shape}\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=182, test_size=0.1, stratify=y)

    print("Fitting model...")
    dtc = Pipeline(steps=[('countvec', CountVectorizer(lowercase=True, stop_words=stopwords.words('english'))), 
                          ('tfidftrans', TfidfTransformer()),
                          ('dtc', DecisionTreeClassifier(max_depth=5, random_state=182))]).fit(X_train, y_train)
    
    print("Model fitted...")

    print("Getting accuracy...")
    print(f"Accuracy: {accuracy_score(y_test, dtc.predict(X_test)) * 100:.4f}%")
    end = time()
    print(f"The script took {end - start} seconds to get ready\n")


    while True:
        statement = [input("Enter a statement: ")]
        predictedEmotion = dtc.predict(statement)
        # print(predictedEmotion)
        predictedEmotion = emotionEncode.inverse_transform([predictedEmotion])[0]
        img = imageData.getImage(predictedEmotion)
        plt.imshow(img)
        plt.title(f"Predicted emotion: {predictedEmotion}")
        plt.show()

if __name__ == '__main__':
    DIR = Path(__file__).parent
    df = pd.read_csv(DIR / "tweet_emotions.csv")
    main(df=df)