import numpy as np
import re


class StringMatching(object):
    """
    Singleton Class which implements BM25 algorithm from pyserini library.
    """
    def __new__(cls, data_handler):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StringMatching, cls).__new__(cls)
            cls.instance.__data = data_handler.retrieve()
            cls.instance.__dataset_name = data_handler.get_dataset_name()
        return cls.instance

    def search(self, keywords):
        keywords_array = keywords.split(" ")
        if len(keywords_array) == 1:
            res_rows = self.__data[self.__data['caption'].str.contains(
                keywords, case=False)]
            # Choose random index out of results
            rand_res_index = np.random.choice(res_rows.shape[0], 1)[0]
            image_name = res_rows.iloc[rand_res_index]
            return image_name
        else:
            #  Number of matches non-sequential
            regex_phrase = ".*".join(
                [f"({key})" for key in keywords_array]) + " "
            data_match_score = self.__data['caption'].apply(
                lambda x: len(re.findall(rf"{regex_phrase}", x)))
            # Get max score rows and choose randomly
            max_score_image_index = data_match_score.idxmax()
            max_score_row = self.__data.iloc[max_score_image_index]
            return max_score_row
