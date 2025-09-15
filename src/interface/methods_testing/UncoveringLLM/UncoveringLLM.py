import torch, torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel



class Analyzer():
    def __init__(self, 
                 tokenizer = "microsoft/graphcodebert-base", 
                 LLM = "./src/interface/methods_testing/UncoveringLLM/model"):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.truncation_side = "left" # ho letto che le ultime riche del codice sono le più importanti

        self.LLM = AutoModel.from_pretrained(LLM, add_pooling_layer=False)
        self.LLM = self.LLM.eval().to(self.DEVICE)



    def _encode_codes(self, original_code: str, 
                    rewrites: list[str]):
        
        '''return CLS value of the orignal code and his rewrites using the encoder'''
        enc = self.tokenizer(
            [original_code] + rewrites,
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
        score = self.detect(original_code = row.get("code"),
                            rewrites=[
                            row.get("code_rewrited"),
                            row.get("code_rewrited2")], 
                            )
        return score



# map function  -> ds3 = ds.map(f, fn_kwargs={"analyzer": analyzer})
def map_functionAnalyzer(ex, analyzer:Analyzer):
    '''how to use this function: ds3 = ds.map(map_functionAnalyzer, fn_kwargs={"analyzer": analyzer})'''
    return {"score": analyzer.detect_dataset(ex)}

