import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import sqlite3
import json
import re
from os.path import exists, dirname, realpath
import io

CURRENT_PATH = dirname(realpath(__file__))
CLUSTER_PATH = '/cs/labs/tomhope/noamdel/medical_search_engine'
DB_PATH = 'search_engine.db'

class DataHandler(object):
    """
    Singleton Class which does all the handling with the dataset.
    """

    def __new__(cls, dataset_name):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DataHandler, cls).__new__(cls)
            cls.instance.dataset_name = dataset_name
            cls.instance.determine_root_dir()
            cls.instance.init_data()
            cls.instance.init_db()
        return cls.instance

    def determine_root_dir(self):
        if not exists(CLUSTER_PATH):
            self.root_dir = CURRENT_PATH
        else:
            self.root_dir = CLUSTER_PATH

    def init_data(self):
        if self.dataset_name == "ROCO":
            self.img_path = f'{self.root_dir}/roco-dataset/data/train/radiology/images'
            self.data_path = f'{self.root_dir}/roco-dataset/data/train/radiology/captions.txt'
            self.data = pd.read_csv(self.data_path, names=["image", "caption"], index_col=False, on_bad_lines='skip', sep="\t")
        elif self.dataset_name == "PMC":
            self.img_path = f'{self.root_dir}/pmc_oa/caption_T060_filtered_top4_sep_v0_subfigures'
            self.data_path = f'{self.root_dir}/pmc_oa/pmc_oa.jsonl'
            self.data = pd.read_json(self.data_path, orient='records', lines=True, typ='frame')
        self.remove_missing_images()
        self.dlinks = self.process_dlinks()

    def remove_missing_images(self):
        rows_to_remove = []
        for idx, row in self.data.iterrows():
            if not exists(self.get_image(row['image'])):
                rows_to_remove.append(idx)
        self.data.drop(rows_to_remove, inplace=True)
        # self.data.to_csv(self.data_path)

    def init_db(self):
        if exists(f'{self.root_dir}{DB_PATH}'):
            return
        self.connect_to_db()
        self.cursor.execute('''CREATE TABLE search_results(
                            keywords TEXT,
                            images TEXT,
                            embeddings BLOB
                            )''')
        self.cursor.execute('''CREATE TABLE favorites(
                            keywords TEXT,
                            images TEXT,
                            embeddings BLOB
                            )''')
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)
        self.close_db()
    
    def process_dlinks(self):
        if self.dataset_name == "ROCO":
            dlinks_csv = pd.read_csv(f'{self.root_dir}/roco-dataset/data/train/radiology/dlinks.txt', names=[
                                    "image", "command", "original_name"], index_col=False, on_bad_lines='skip', sep="\t")
            dlinks_csv['command'] = dlinks_csv['command'].apply(
                lambda x: re.findall(r'\/(PMC[0-9]+)\.tar\.gz', x))
            dlinks_csv.drop(columns=['original_name'])

        elif self.dataset_name == "PMC":
            dlinks_csv = self.data['image'].apply(lambda x: re.search("PMC[0-9]+", x).group())
        return dlinks_csv
    
    def connect_to_db(self):
        self.connection = sqlite3.connect(f'{self.root_dir}{DB_PATH}', detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()
    
    def close_db(self):
        self.connection.commit()
        self.cursor.close()

    def preview(self):
        return self.data.head()
    
    def retrieve(self):
        return self.data

    def get_image(self,image:str):
        # image_path = "{imgs_path}/{img_name}".format(imgs_path=self.img_path,img_name=image)
        image_path = "{imgs_path}/{img_name}.jpg".format(imgs_path=self.img_path,img_name=image)
        return image_path
        # return plt.imread(image_path)
    
    def get_image_key(self, image_name):
        keys = self.dlinks.loc[self.dlinks['image'] == image_name]['command'].item()
        if len(keys) > 0:
            return keys[0]
        else:
            return ''

    def get_image_caption(self, image_name: str) -> str:
        data_row = self.data.loc[self.data['image'] == image_name]
        return data_row.iloc[0]['caption']
    
    def get_favorites(self, search_input):
        self.connect_to_db()
        db_field = "keywords" if type(search_input) is str else "embeddings"
        self.cursor.execute(f"SELECT images FROM favorites WHERE {db_field} = ?;", (search_input,))
        favorite_images = self.cursor.fetchone()
        res = []
        if favorite_images:
            res = json.loads(favorite_images[0])
        self.close_db()
        return res
    
    def show_image(self,image):
        pass
        # plt.imread(image)
        # plt.imshow(image)
        # plt.show()

    def adapt_array(self, data):
        out = io.BytesIO()
        np.save(out, data)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def save_to_db(self, table_name, search_input, data_to_save):
        self.connect_to_db()
        if isinstance(search_input, np.ndarray):
            db_field = "embeddings"
        else:
            db_field = "keywords"
        curr_results = [data_to_save] if type(data_to_save) == str or type(data_to_save) == np.str_ else list(data_to_save)
        self.cursor.execute(f"SELECT images FROM {table_name} WHERE {db_field} = ?;", (search_input,))
        prev_results = self.cursor.fetchone()
        if prev_results is not None:
            prev_results = json.loads(prev_results[0])
            curr_results = list(set(curr_results + prev_results))
            self.cursor.execute(f"UPDATE {table_name} SET images = ? WHERE {db_field} = ?;", (json.dumps(curr_results), search_input))
        else:
            self.cursor.execute(f"INSERT INTO {table_name}({db_field}, images) VALUES(?,?);", (search_input, json.dumps(curr_results)))
        self.close_db()
    
    def save_favorite_image(self, search_input, img):
        print(f"@@@ img type: {type(img)}")
        self.save_to_db('favorites', search_input, img)

    def save_search_results(self, search_input, imgs):
        self.save_to_db('search_results', search_input, imgs)

    def get_from_db(self, table_name, search_input):
        db_field = "keywords" if type(search_input) is str else "embeddings"
        self.cursor.execute(f"SELECT images FROM {table_name} WHERE {db_field} = ?;", (search_input,))
        return self.cursor.fetchall()

    def get_dataset_name(self):
        return self.dataset_name