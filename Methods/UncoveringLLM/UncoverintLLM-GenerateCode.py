import os
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# so we don't have path probelms (obtain this script's path)
current_file = Path(__file__).resolve()
current_folder = current_file.parent


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PLM_NAME = "microsoft/graphcodebert-base"  # same training backbone (GraphCodeBert)
CKPT_DIR =  Path(os.path.join(current_folder, "UncoveringLLMGeneratedCode/drive/simcse/ckpt_gcb4_full/best_model/")) # trained model



tokenizer = AutoTokenizer.from_pretrained(PLM_NAME)
encoder   = AutoModel.from_pretrained(CKPT_DIR).to(DEVICE).eval() 




###
@torch.no_grad() # disable gradient


#TODO: ultimi blocchi semantici per rientrare in lunghezza 

def encode_codes(codes, max_len=512, batch_size=32):
    '''
    codes: total number of codes
    batch_size: number of codes in a batch
    '''
    vecs = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = encoder(**enc).last_hidden_state            # [B, L, Embedding]
        cls = out[:, 0, :]                                # first token: [CLS]
        cls = F.normalize(cls, p=2, dim=-1)               # SimCSE: L2 normalization
        vecs.append(cls.cpu())
    return torch.cat(vecs, dim=0)                         # [N, H]



# best number of rewrites is 4
def detect(code, rewrites, threshold=0.90):
    
    if len(rewrites)!= 4:
        print(f"Number of rewrites is not 4 but:{len(rewrites)}")

    embs = encode_codes([code, *rewrites])
    e0, eR = embs[0:1], embs[1:]
    sims = F.cosine_similarity(eR, e0.expand_as(eR), dim=-1).numpy()
    score = float(sims.mean()) if len(sims) else float("nan")
    return score, (score >= threshold), sims


