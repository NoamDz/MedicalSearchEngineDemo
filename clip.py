
from open_clip import create_model_from_pretrained, get_tokenizer
# from transformers import CLIPProcessor, CLIPModel
from os import path, environ
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from data import DataHandler
from datasets import Dataset, load_from_disk
environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



CURRENT_PATH = path.dirname(path.realpath(__file__))
CLUSTER_PATH = '/cs/labs/tomhope/noamdel/medical_search_engine'
BIOMEDCLIP = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
ENCODING_DIMENSION = 512
TEXT_TEMPLATE = 'this is a photo of '
TOP_K = 5


class ClipModel(object):
    """
    Singleton Class which utelized BioMedClip NN.
    """

    def __init__(self, data_handler):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model, self.preprocessor = create_model_from_pretrained(BIOMEDCLIP, device=self.device)
        self.tokenizer =  get_tokenizer(BIOMEDCLIP)
        self.data_handler = data_handler
        self.set_up()
        # cls.instance.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # cls.instance.preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # cls.instance.data = data_handler.retrieve()

    def set_up(self):
        self.set_algorithm_paths()
        self.load_embeddings()

    def create_text_embeddings(self):
        data = self.data_handler.retrieve()
        if path.exists(self.torch_text_embeddings):
            embeddings = torch.load(self.torch_text_embeddings, map_location=self.device)
        else:
            self.model.to(self.device)
            self.model.eval()
            embeddings = torch.empty(size=(0, ENCODING_DIMENSION)).to(self.device)
            batch_size = 50
            batch_num = data.shape[0] // batch_size
            with torch.no_grad():
                for i, batch in enumerate(tqdm(np.array_split(data.caption.values, batch_num))):
                    tokens = self.tokenizer([TEXT_TEMPLATE + caption for caption in batch]).to(self.device)
                    text_features = self.model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    embeddings = torch.cat((embeddings, text_features),0)

            torch.save(embeddings, self.torch_text_embeddings)
        self.text_embeddings = Dataset.from_dict({"embeddings": embeddings.to(torch.device('cpu')), "image":data[['image']].values})
        self.text_embeddings.save_to_disk(self.text_embeddings_path)
        self.text_embeddings.add_faiss_index(column="embeddings")
        self.text_embeddings.save_faiss_index("embeddings", self.caption_index_path)

    def create_image_embeddings(self):
        data = self.data_handler.retrieve()
        if path.exists(self.torch_embeddings):
            embeddings = torch.load(self.torch_embeddings, map_location=self.device)
        else:
            self.model.to(self.device)
            self.model.eval()
            img_paths = [self.data_handler.get_image(img_name) for img_name in data.image]
            embeddings = torch.empty(size=(0, ENCODING_DIMENSION)).to(self.device)
            batch_size = 50
            batch_num = data.shape[0] // batch_size
            with torch.no_grad():
                for i, batch in enumerate(tqdm(np.array_split(img_paths, batch_num))):
                    imgs = torch.stack([self.preprocessor(Image.open(img_path)) for img_path in batch]).to(self.device)
                    image_features = self.model.encode_image(imgs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    embeddings = torch.cat((embeddings, image_features),0)

            torch.save(embeddings, self.torch_embeddings)
        self.embeddings = Dataset.from_dict({"embeddings": embeddings.to(torch.device('cpu')), "image":data[['image']].values})
        self.embeddings.save_to_disk(self.embeddings_path)
        self.embeddings.add_faiss_index(column="embeddings")
        self.embeddings.save_faiss_index("embeddings", self.index_path)

    def load_embeddings(self):
        if path.exists(self.text_embeddings_path):
            self.text_embeddings = load_from_disk(self.text_embeddings_path)
            self.text_embeddings.load_faiss_index('text_embeddings', self.index_path)
        else:
            self.create_text_embeddings()
        
        if path.exists(self.embeddings_path):
            self.embeddings = load_from_disk(self.embeddings_path)
            self.embeddings.load_faiss_index('embeddings', self.index_path)
        else:
            self.create_image_embeddings()

    def set_algorithm_paths(self):
        embeddings_dir = CLUSTER_PATH
        if not path.exists(CLUSTER_PATH):
            embeddings_dir = CURRENT_PATH
        dataset_name = self.data_handler.get_dataset_name()
        # self.embeddings_path = f'{CURRENT_PATH}/embeddings/{dataset_name}.pkl'
        self.embeddings_path = f'{embeddings_dir}/embeddings/{dataset_name}.hf'
        self.text_embeddings_path = f'{embeddings_dir}/embeddings/captions_{dataset_name}.hf'
        self.index_path = f'{embeddings_dir}/embeddings/{dataset_name}.faiss'
        self.caption_index_path = f'{embeddings_dir}/embeddings/captions_{dataset_name}.faiss'
        self.torch_embeddings = f'{embeddings_dir}/embeddings/embedding_torch_file.pt'
        self.torch_text_embeddings = f'{embeddings_dir}/embeddings/text_embedding_torch_file.pt'
        # self.indices_path = f'{CURRENT_PATH}/indexes/{dataset_name}'

    def search_image(self, query):
        with torch.no_grad():
            image = self.preprocessor(Image.open(query)).unsqueeze(0).to(self.device)
            query_imbeddings = self.model.encode_image(image).cpu().detach().numpy()
            scores, samples = self.embeddings.get_nearest_examples("embeddings", query_imbeddings, k=5)
            return np.squeeze(np.array(samples["image"]).T), query_imbeddings

    def search_text(self, query):
        with torch.no_grad():
            tokens = self.tokenizer(query).to(self.device)
            query_imbeddings = self.model.encode_text(tokens).cpu().detach().numpy()
            scores, samples = self.text_embeddings.get_nearest_examples("text_embeddings", query_imbeddings, k=5)
            return np.squeeze(np.array(samples["image"]).T), query_imbeddings




if __name__=="__main__":
    data_handler = DataHandler('ROCO')
    clip = ClipModel(data_handler)