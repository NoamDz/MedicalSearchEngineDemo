
# 🩺🔍 MedClip Search – Multimodal Medical Image & Text Retrieval Engine

> A Streamlit-powered demo that lets radiologists and physicians **search millions of peer-reviewed cases by either image _or_ free-text**, powered by a BiomedCLIP backbone and BM25 ranking.  
> Built at **Tom Hope Lab (guided by Nir Mazor)**.

![screenshot placeholder](docs/demo_screenshot.png)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)](https://streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ Key Features
- **Bidirectional Search** – drop an image *or* type a pathology; get the closest cases instantly.  
- **Two Large Open Datasets** – Radiology Objects in Context (ROCO, ≈80 k pairs) & PMC-OA (≈1.6 M pairs).  
- **Domain-tuned Embeddings** – uses **BiomedCLIP** image encoder; captions indexed with **BM25 (Pyserini)**.  
- **Interactive Feedback Loop** – like / dislike results, build a favourites tray, and refine on the fly.  
- **Lightweight Storage** – embeddings saved as JSON blobs in SQLite; no extra infra needed.  
- **Cluster-Ready** – ships with pre-computed indices & embeddings for the lab’s GPU cluster.

---

## 📑 Table of Contents
1. [Quick Start](#-quick-start)  
2. [Datasets & Pre-processing](#-datasets--pre-processing)  
3. [Index & Embedding Generation](#-index--embedding-generation)  
4. [Running the App](#-running-the-app)  
5. [System Architecture](#-system-architecture)  
6. [Roadmap](#-roadmap)  
7. [Contributing](#-contributing)  
8. [Citation](#-citation)  
9. [License](#-license)

---

## 🚀 Quick Start

### Option A – Lab Cluster *(Recommended)*
```bash
ssh <your-user>@<cluster-host>
git clone https://github.com/<you>/medclip-search.git
cd medclip-search
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run gui.py              # indices & embeddings already on the cluster
````

### Option B – Local Setup

1. Complete steps 1-5 above on your workstation.
2. Download a dataset (ROCO or PMC-OA) into `data/`.
3. Edit `gui.py` → `DATASET = "ROCO"` *or* `"PMC"`.
4. Prepare folders:

   ```bash
   mkdir -p collections/ROCO  index/ROCO  embeddings
   ```
5. Build indices & embeddings:

   ```bash
   python clip.py                      # may take hours on CPU – use GPU if possible
   ```
6. Launch:

   ```bash
   streamlit run gui.py
   ```

---

## 🗂️ Datasets & Pre-processing

| Dataset    | Pairs  | Modality Notes                                              |
| ---------- | ------ | ----------------------------------------------------------- |
| **ROCO**   | ≈80 k  | Figure-caption pairs from PubMed Central radiology articles |
| **PMC-OA** | ≈1.6 M | Broad biomedical literature, open-access                    |

For each pair we store: `id`, `image → PNG`, `caption`, and BiomedCLIP **768-D** image vector.
Captions are written to a **.jsonl** “collection” file for BM25 indexing (via Pyserini).

---

## 🏗️ Index & Embedding Generation

`clip.py` does the heavy lifting:

1. **Image Encoder** – BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`).
2. **Vector Store** – 768-D vectors → JSON → SQLite.
3. **Text Index** – Pyserini builds a Lucene BM25 index on captions.

To reclaim disk space you can safely delete raw images after vectors are stored.

---

## 🖥️ Running the App

* **Home page** – drop an image or type text; results stream in order of cosine/BM25 score.
* **Result cards** – hover for caption; click for full-size figure; ⭐ to favourite.
* **Feedback** – 👎 on a card triggers re-ranking based on your ⭐ tray.

> ⚠️ The Streamlit session keeps vectors in RAM (\~2 GB for ROCO; \~30 GB for full PMC).
> Use `--server.maxMessageSize` flag if you hit Protobuf size limits.

---

## 🧬 System Architecture

```text
┌────────────┐      text query       ┌──────────────┐
│ Streamlit  │ ────────────────────▶│   BM25 IDX   │
│   Front-end│                      └──────────────┘
│ (gui.py)   │
│            │      image query      ┌──────────────┐
│            │ ────────────────────▶ │ BiomedCLIP   │ 768-D
│            │                      │   Encoder     │
│            │                      └──────────────┘
│ ────────────────────────────────────────────────────
│            │ top-k ids / vectors  ┌──────────────┐
│            │ ◀─────────────────── │ SQLite + NNS │
└────────────┘                      └──────────────┘
```

* **Nearest-Neighbour Search** – cosine similarity with a custom NumPy/FAISS fallback.
* **Re-ranking** – combines BM25 or cosine score with user feedback Weight-of-Evidence.

---

## 🛣️ Roadmap

* [ ] **Multi-modal Fusion** – cross-encoder to blend text + image queries.
* [ ] **DICOM Drag-and-Drop** – parse `.dcm` stacks directly.
* [ ] **GPU-accelerated FAISS IVF** for million-scale latency < 50 ms.
* [ ] **Clinician Study** – measure diagnostic time-saving vs. PubMed search.

---

## 🤝 Contributing

Pull requests are welcome—feel free to open an issue first to discuss major changes.
Please run `black` + `isort` before committing.

---

## 📜 Citation

If you use this code or our pre-computed indices, please cite:

```bibtex
@software{delbari_2025_medclip,
  author       = {Noam Delbari},
  title        = {MedClip Search Engine},
  year         = {2025},
  url          = {https://github.com/<you>/medclip-search}
}
```

For the underlying models & datasets, see `docs/references.bib`.

---

## ⚖️ License

**MIT** – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

This project was developed as part of my BSc research in the **Tom Hope Lab** (Hebrew University of Jerusalem). Huge thanks to *Nir Mazor* for guidance and to the lab team for GPU compute and feedback.

```

---
