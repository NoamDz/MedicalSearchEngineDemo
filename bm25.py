
from pyserini.search.lucene import LuceneSearcher
from os import path, system
import json

CURRENT_PATH = path.dirname(path.realpath(__file__))
CLUSTER_PATH = '/cs/labs/tomhope/noamdel/medical_search_engine'
JSON_COLLECTION_KEYS = ["id", "contents"]
TOP_K = 5


class BM25Algorithm(object):
    """
    Singleton Class which implements BM25 algorithm from pyserini library.
    """

    def __new__(cls, data_handler):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BM25Algorithm, cls).__new__(cls)
            cls.instance.__data = data_handler.retrieve()
            cls.instance.set_up(data_handler.get_dataset_name())
        return cls.instance

    def set_up(self, dataset_name):
        # Add directives "collections" and "indexes"
        self.set_algorithm_paths(dataset_name)
        self.create_collection_from_data()
        self.create_index_from_collection()
        self.searcher = LuceneSearcher(self.indices_path)

    def set_algorithm_paths(self, dataset_name):
        data_path = CLUSTER_PATH
        if not path.exists(data_path):
            data_path = CURRENT_PATH
        self.collections_path = f'{data_path}/collections/{dataset_name}'
        self.collection_path = f'{self.collections_path}/{dataset_name}.jsonl'
        self.indices_path = f'{data_path}/indexes/{dataset_name}'

    def create_collection_from_data(self):
        """
        Converts data to JSONL collection for pyserini index.
        """
        if path.exists(self.collection_path):
            print(f"@@@ collection path: ")
            return
        with open(self.collection_path, "w", encoding='utf-8', newline='\n') as file:
            for index, row in self.__data.iterrows():
                row_dict = dict(zip(JSON_COLLECTION_KEYS, row.to_list()[0:2]))
                json_str = json.dumps(row_dict)
                file.write(json_str + '\n')
        file.close()

    def create_index_from_collection(self):
        """
        Creates and Index for a collection,
        """
        if path.exists(self.indices_path):
            return
        cmd = f'python3 -m pyserini.index.lucene ' + \
            f'-collection JsonCollection ' + \
            f'-input {self.collections_path} ' +\
            f'-index {self.indices_path} ' +\
            f'-generator DefaultLuceneDocumentGenerator ' + \
            f'-threads 1 '
        system(cmd)

    def search(self, key_words):
        hits = self.searcher.search(key_words)
        top_k_images = [value.docid for value in hits[:TOP_K]]
        top_k_scores = [value.score for value in hits[:TOP_K]]
        return top_k_images, top_k_scores
