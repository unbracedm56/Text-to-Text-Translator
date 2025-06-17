import torch
from torch import nn
import sentencepiece as spm
import pandas as pd
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp_en = spm.SentencePieceProcessor()
sp_en.load('tokenizer\spm_en.model')

sp_es = spm.SentencePieceProcessor()
sp_es.load('tokenizer\spm_es.model')

vocab_size_en = sp_en.get_piece_size()
vocab_size_es = sp_es.get_piece_size()
EMBED_SIZE = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 5

encoder = utils.Encoder(vocab_size_en, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
decoder = utils.Decoder(vocab_size_es, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

sos_id = sp_es.bos_id()
eos_id = sp_es.eos_id()

encoder.load_state_dict(torch.load(f="model\lstm_encoder_final_17.pth", map_location=torch.device(device)))
decoder.load_state_dict(torch.load(f="model\lstm_decoder_final_17.pth", map_location=torch.device(device)))

def for_translation(sentence):
    translation = utils.translate_sentence(sentence, encoder, decoder, sp_en, sp_es, sos_id, eos_id, device)

    clean_text = translation.split('⁇')[0].strip()
    return clean_text