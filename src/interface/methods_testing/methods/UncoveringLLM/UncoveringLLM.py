import torch, torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import os
from pathlib import Path


class Analyzer():
    def __init__(self, code_conlum = 'cleared_code',
                 tokenizer = "microsoft/graphcodebert-base", 
                 LLM = None):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.code_conlum = code_conlum

        # Build absolute path to model directory
        if LLM is None:
            HERE = Path(__file__).resolve().parent
            LLM = os.path.abspath(os.path.join(HERE, "model"))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.truncation_side = "left" # read that the last lines of code are the most important

        self.LLM = AutoModel.from_pretrained(LLM, add_pooling_layer=False)
        self.LLM = self.LLM.eval().to(self.DEVICE)



    def _encode_codes(self, original_code: str, 
                    rewrites: list[str]):
        
        '''return CLS value of the orignal code and his rewrites using the encoder'''
        # Filter out None or empty strings from rewrites
        valid_rewrites = [r for r in rewrites if r]
        
        if not original_code or not valid_rewrites:
            raise ValueError("original_code and at least one valid rewrite are required")
        
        enc = self.tokenizer(
            [original_code] + valid_rewrites,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )


        enc = {k: v.to(self.DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = self.LLM(**enc).last_hidden_state   # [B, L, H]
            cls = out[:, 0, :]                       # [B, H]
            cls = F.normalize(cls, p=2, dim=-1)

        orig_emb = cls[0]         # [H]
        rewrite_embs = cls[1:]    # [R, H]
        return orig_emb, rewrite_embs




    def detect(self, original_code: str, 
               rewrites: list[str]) -> float:
        
        ''' return the similarity score betwen original code and his rewrites '''

        e0, eR = self._encode_codes(original_code = original_code, 
                                    rewrites =rewrites)  # e0: [H], eR: [R, H]
        sims = F.cosine_similarity(
                    eR, e0.unsqueeze(0).expand_as(eR), dim=-1 
                ).cpu().numpy()
        score = float(sims.mean()) if len(sims) else float("nan")
        return score
    

    def detect_dataset(self, row) -> float:
        # Validate that all required columns exist and are not None
        original_code = row.get(self.code_conlum)
        code_rewrited = row.get("code_rewrited")
        code_rewrited2 = row.get("code_rewrited2")
        
        # Check for None or empty values
        if not original_code or not code_rewrited or not code_rewrited2:
            return float("nan")
        
        score = self.detect(original_code=original_code,
                            rewrites=[code_rewrited, code_rewrited2])
        return score



# map function  -> ds3 = ds.map(f, fn_kwargs={"analyzer": analyzer})
def map_functionAnalyzer(ex, analyzer:Analyzer):
    '''how to use this function: ds3 = ds.map(map_functionAnalyzer, fn_kwargs={"analyzer": analyzer})'''
    return {"score": analyzer.detect_dataset(ex)}

