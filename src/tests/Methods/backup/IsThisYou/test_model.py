from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, os

ckpt = "Salesforce/codet5p-770m"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
         else torch.float16 if torch.cuda.is_available()
         else torch.float32)

tokenizer = AutoTokenizer.from_pretrained(ckpt)

model = AutoModelForSeq2SeqLM.from_pretrained(
    ckpt,
    low_cpu_mem_usage=True,     # evita doppie copie
    device_map="auto",          # invia direttamente ai device
    torch_dtype=dtype,
    offload_state_dict=True,    # <-- chiave: non tiene lo state_dict in RAM
    offload_folder="offload",   # cartella per i file temporanei
    # niente use_safetensors qui (il repo ha solo .bin)
)

inputs = tokenizer("def print_hello_world():<extra_id_0>", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.inference_mode():
    out = model.generate(**inputs, max_length=20)
print(tokenizer.decode(out[0], skip_special_tokens=True))
