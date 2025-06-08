Medical Search engine project:

This project is a web search engine where radiolosits can search for cases they're intersted in by entering description of a pathology they're interested in or insert an image they would like to validate its pathology and find similiar cases.


How to use it?

Using Lab's Cluster (*recommended):
1. Connect to your user at the cluster
2. Clone this repository using 'git clone' command
3. Create new virtual environment: virtualenv venv
4. Activate environment: source venv/bin/activate
5. Install the requirements.txt file: python3 -m pip install -r requirements.txt
6. Run the following command: 'streamlit run gui.py'
* All datasets and encoded data already exist over in my user

Using you local env:
1. Run steps 1-5 from Cluster instrcutions.
2. Clone either ROCO or PMC OA dataset to your project directory
3. In gui.py change line 8 to the dataset you're using 
4. Create 'collections' directory in your project and create a subdirectory with 'ROCO' or 'PMC' depending on you dataset.
5. Create 'indexs' directory in your project and create a subdirectory with 'ROCO' or 'PMC' depending on you dataset.
6. Create 'embeddings' folder for both the torch embedding files - 'embedding_torch_file.pt' for images and 'text_embedding_torch_file.pt' for captions (will be created in the next step)
7. Run clip.py file: python3 clip.py, this will create both the index for your dataset and embeddings
8. Now you're all set, run the command: 'streamlit run gui.py'