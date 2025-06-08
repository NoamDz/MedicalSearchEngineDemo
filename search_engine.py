from data import DataHandler
from bm25 import BM25Algorithm
from clip import ClipModel
import numpy as np
from string_matching import StringMatching

class SearchEngine(object):
    """
    Singleton class to represent the search engine with an API to main app.
    """
    last_search = None

    def __new__(cls: object, dataset_name: str) -> object:
        if not hasattr(cls, 'instance'):
            cls.instance = super(SearchEngine, cls).__new__(cls)
            cls.instance.data_handler = DataHandler(dataset_name)
            data_handler = cls.instance.data_handler
            cls.instance.text_model = BM25Algorithm(data_handler)
            cls.instance.image_model = ClipModel(data_handler)
        return cls.instance

    def search_by_keyword(self, key_words: str):
        results, scores = self.text_model.search(key_words)
        print(f"Results: {results}")
        if self.last_search is not None and self.last_search == key_words:
            return results, scores
        self.last_search = key_words
        self.data_handler.save_search_results(key_words, results)
        return results, scores

    def search_by_image(self, img):
        results, embeddings = self.image_model.search_image(img)
        if self.last_search is not None and not np.array_equal(self.last_search, embeddings):
            return results
        self.last_search = embeddings
        self.data_handler.save_search_results(embeddings, results)
        return results

    def save_image(self, image_name , ok):
        self.data_handler.save_favorite_image(self.last_search, image_name)

    def test_class(self):
        print(self.data_handler.preview())

    def show_image(self, image_name):
        self.data_handler.show_image(image_name)

    def get_image(self, img_obj):
        image_name = img_obj
        if type(image_name) != str and type(image_name) != np.str_:
            image_name = image_name['image']
        return self.data_handler.get_image(image_name)

    def get_image_caption(self, img_obj):
        if type(img_obj) != str and type(img_obj) != np.str_:
            return img_obj['caption']
        return self.data_handler.get_image_caption(img_obj)

    def get_image_link(self, image_name):
        image_key = self.data_handler.get_image_key(image_name)
        return f'https://www.ncbi.nlm.nih.gov/pmc/articles/{image_key}'

    def get_favorites(self):
        favorites =  self.data_handler.get_favorites(self.last_search)
        return favorites

    def get_highest_score_image(self, images):
        max_score = 0
        new_image = None
        for image_name in images:
            image_caption = self.data_handler.get_image_caption(image_name)
            results, scores = self.search_by_keyword(image_caption)
            if len(results) > 1 and scores[1] > max_score:
                new_image = results[1]
        return new_image

    def find_new_image(self, key_words, search_result, image_to_replace):
        favorite_images = self.get_favorites(key_words)
        if image_to_replace in favorite_images:
            favorite_images.remove(image_to_replace)
        new_image = self.get_highest_score_image(favorite_images)
        if new_image:
            return new_image
        ## Fallback to choose between all search results:
        search_result.remove(image_to_replace)
        return self.get_highest_score_image(search_result)


if __name__ == "__main__":
    search_engine = SearchEngine("clip", "PMC")
    # results = search_engine.search_by_keyword(
    #     "Three months later, the same patient in Fig. 1 experienced severe, recurrent abdominal pain with laboratory evidence of systemic inflammation (14.000/mmc leukocytes, 34\u00a0mg/L C-reactive protein). On further questioning, he admitted discontinuation of antacids and H2-blocker medications, and nonsteroidal anti-inflammatory drugs (NSAIDs) intake. Repeated CT revealed increased hypoattenuating mural thickening (*) with mucosal enhancement (thin arrows) of the pylorus and proximal duodenum which spared the anterior aspect, worsened periduodenal inflammatory changes (+), and development of a 2-cm roundish posterior deep ulcer (arrows). Upper digestive endoscopy confirmed a non-malignant retropyloric peptic ulcer. The patient ultimately did well on proton-pump inhibitors (PPI) and anti-Helicobacter pylori triple therapy [Adapted from Open Access ref. no")
    # imgs = [search_engine.get_image(img_name) for img_name in results]
    # captions = [search_engine.get_image_caption(img_name) for img_name in results]
    # for i in range(len(imgs)):
    #     print(f'Caption: {captions[i]}')
    #     print(f'Image name: {imgs[i]}')
    #     search_engine.show_image(imgs[i])
