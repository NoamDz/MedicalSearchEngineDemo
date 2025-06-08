import streamlit as st
import time
from search_engine import SearchEngine
import json

RETRIEVED_IMGAGES_NUMBER = 5
class SimpleGui(object):
    search_engine = SearchEngine("PMC")

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SimpleGui, cls).__new__(cls)
        return cls.instance
    
    @staticmethod
    def search_result_exist(results):
        return results is not None and len(results) > 0
    
    def discription_generator(self):
        for word in st.session_state.query_description.split(" "):
            yield word
            time.sleep(0.01)
    
    def display_search_results(self):
        with st.container(border=True):
            left_co, last_co = st.columns(2)
            with left_co:
                generator = self.discription_generator()
                st.write_stream(generator)
            with last_co:
                st.image(st.session_state.query_image, width=300, caption="Potential pathologies: " + ",".join(st.session_state.query_label))
        st.subheader("The following images found to be most fitting for your case:")

        for index in range(RETRIEVED_IMGAGES_NUMBER):
            with st.container(border=True):
                left_co, cent_co, right_co, = st.columns(3)
                with cent_co:
                    st.image(st.session_state.retrieved_imgs[index], width=300, caption=st.session_state.retrieved_captions[index])

    def describe_query(self, retrieved_imgs):
        return "Magnetic Resonance Imaging (MRI) is a medical imaging technique that has revolutionized the way we diagnose and understand various medical conditions. It uses strong magnetic fields and radio waves to generate detailed images of the organs and tissues within the body. Unlike X-rays and CT scans, MRI does not use ionizing radiation, making it a safer option for imaging, especially for repeated studies. The development of MRI began in the early 20th century, with significant contributions from several scientists. In 1937, Isidor Rabi discovered nuclear magnetic resonance (NMR), for which he won the Nobel Prize in Physics in 1944. The application of NMR to imaging was first proposed by Paul Lauterbur in 1973, who shared the 2003 Nobel Prize in Physiology or Medicine with Peter Mansfield for their discoveries concerning MRI. An MRI machine consists of a large, cylindrical magnet. The patient lies on a movable bed that slides into the magnet. Once inside, the machine generates a strong magnetic field that aligns the protons in the body. Radiofrequency (RF) pulses are then sent through the body, causing these protons to produce signals that are detected by the MRI scanner. These signals are used to create images of the body's internal structures. MRI is particularly useful for imaging the brain, spinal cord, and nerves, as well as muscles, joints, and ligaments. It provides a high level of detail that is invaluable in diagnosing a range of conditions. For instance, MRI can detect brain"
    
    def get_images_and_captions(self, imgs_names):
        imgs = [self.search_engine.get_image(img_name) for img_name in imgs_names]
        captions = [self.search_engine.get_image_caption(img_name) for img_name in imgs_names]
        return imgs, captions

    def generate_response(self):
        imgs_names = self.search_engine.search_by_image(st.session_state["query_image"])
        retrieved_imgs, retrieved_captions = self.get_images_and_captions(imgs_names)
        query_description = self.describe_query(retrieved_imgs)
        return retrieved_imgs, retrieved_captions, query_description
        
    def set_state_search_input(self, input):
        if 'search_input' not in st.session_state or st.session_state.search_input != input:
            st.session_state.search_input = input

    def results_exist(self):
        if "retrieved_imgs" not in st.session_state or "retrieved_captions" not in st.session_state or "query_description" not in st.session_state:
            return False
        return True
    
    def is_valid_submition(self):
        if not st.session_state.query_image or not st.session_state.query_label:
            st.markdown(''':red[Missing input - Make sure to upload an image and select patholgy]''')
            return False
        return True

    def read_labels(self):
        with open('disease_dict_mesh_keys.json', 'r') as file:
            data_list = json.load(file)
            labels = data_list["keys"]
            return labels

    def run(self):
        st.set_page_config(page_title="SearchEngine", page_icon=":tada:", layout="wide")
        st.title("Medical search engine")
        st.caption('''This apps provide meaningful insights on medical cases and displays similar pathologies''')
        with st.sidebar:
            labels = self.read_labels()
            st.multiselect("Select potential pathology", labels, max_selections=1, key='query_label')
            st.file_uploader("Upload an image", key="query_image")
            submitted = st.button('Search')
        if submitted and self.is_valid_submition():
            st.session_state.retrieved_imgs, st.session_state.retrieved_captions, st.session_state.query_description = self.generate_response()
            self.display_search_results()


if __name__ == "__main__":
    gui = SimpleGui()
    gui.run()
