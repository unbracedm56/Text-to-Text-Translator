# 🧠 English-to-Spanish Text Translator

This project is a **text-to-text machine translation system** that translates English sentences to Spanish using an **LSTM-based encoder-decoder architecture**. It uses **custom SentencePiece tokenizers**, and the model weights are hosted on the Hugging Face Hub. A minimal **Streamlit web interface** is provided for interaction.

---

## 🧠 Model Details

- **Architecture**: LSTM encoder-decoder
- **Embedding Size**: 512  
- **Hidden Size**: 1024  
- **Layers**: 5  
- **Tokenization**: SentencePiece (subword units)
- **Training**: Done offline; only inference is supported here

Model weights are downloaded from [Hugging Face Hub](https://huggingface.co/unbracedm56/lstm_models):

- `lstm_encoder_final_17.pth`
- `lstm_decoder_final_17.pth`

---

## 🌐 Running the App

### 1. Clone the Repository
```bash
git clone https://github.com/unbracedm56/Text-to-Text-Translator.git
cd Text-to-Text-Translator
```

### 2. Setup Environment
Install the dependencies:
```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root and add your Hugging Face token:
```ini
HF_TOKEN=your_huggingface_token
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## ✨ Features

- LSTM encoder-decoder based translation.
- Tokenized using SentencePiece for better generalization.
- Clean translation output using heuristic filtering.
- Model weights hosted on Hugging Face for easy portability.
- Simple and intuitive Streamlit UI.

---

## 🔐 Environment Variables

To run this project, you need to define the following environment variable:

| Variable   | Description                                                  |
|------------|--------------------------------------------------------------|
| `HF_TOKEN` | Your Hugging Face access token (used to download model files) |

---

## 🙋‍♂️ Acknowledgements
- Hugging Face Hub – for model hosting
- PyTorch – deep learning framework
- SentencePiece – subword tokenizer
- Streamlit – frontend UI framework

---

## 🤝 Contributions
Feel free to open issues or pull requests to improve the model or UI.

---
