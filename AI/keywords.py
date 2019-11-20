import gensim
from gensim.summarization import keywords
class KeyWordsExtractor:

        def __init__(self,ratio=0.6):
            self.ratio = ratio
        
        def generate_keywords(self,text):
            words = keywords(text,self.ratio)
            return words

if __name__ == "__main__":

    keyword = KeyWordsExtractor()
    keywords = keyword.generate_keywords("My family. There are four people in my family: my parents, my younger brother and I. My father is 46. He is a worker in a company which is near my house. He likes talking with his friends or watching TV in his free time However, my father is")
    print(keywords)