import torch
from torch import nn

class Encoder(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)

  def forward(self, x):
    embedded = self.embedding(x)
    outputs, (hidden, cell) = self.lstm(embedded)
    return outputs, hidden, cell

class Decoder(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self, x, hidden, cell):
    x = x.unsqueeze(1)
    out = self.embedding(x)
    out, (hidden, cell) = self.lstm(out, (hidden, cell))
    out = self.fc(out)
    return out, hidden, cell

def add_sos_eos(tensor_batch, sos_token_id, eos_token_id, pad_token_id, device):
    batch_size, seq_len = tensor_batch.size()
    new_input = torch.full((batch_size, seq_len + 1), pad_token_id, dtype=torch.long, device=device)
    new_input[:, 0] = sos_token_id
    new_input[:, 1:] = tensor_batch

    new_target = torch.full((batch_size, seq_len + 1), -100, dtype=torch.long, device=device)
    new_target[:, :-1] = tensor_batch
    new_target[:, -1] = eos_token_id

    return new_input, new_target

def translate_sentence(sentence, encoder, decoder, sp_en, sp_es, sos_id, eos_id, device, max_len=50):
  encoder.eval()
  decoder.eval()

  input_tensor = torch.tensor([sp_en.encode(sentence, out_type=int)], dtype=torch.long).to(device)
  with torch.inference_mode():
    _, encoder_hidden, encoder_cell = encoder(input_tensor)

  decoder_input = torch.tensor([sos_id], dtype=torch.long).to(device)

  translated_tokens = []

  hidden, cell = encoder_hidden, encoder_cell

  for _ in range(max_len):
    with torch.inference_mode():
      output, hidden, cell = decoder(decoder_input, hidden, cell)
      output = output.squeeze(1)
      next_token = output.argmax(1)

    if next_token.item() == eos_id:
      break

    translated_tokens.append(next_token.item())
    decoder_input = next_token

  translation = sp_es.decode(translated_tokens)
  return translation