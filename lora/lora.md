### Setting


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cude" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model

model_name = 'WhitePeak/bert-base-cased-Korean-sentiment'

tokenizer, model = load_model(model_name)
model = model.to(device)
```


    tokenizer_config.json:   0%|          | 0.00/367 [00:00<?, ?B/s]


    c:\Users\BizSpring\Desktop\lora\.venv\lib\site-packages\huggingface_hub\file_download.py:142: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\BizSpring\.cache\huggingface\hub\models--WhitePeak--bert-base-cased-Korean-sentiment. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    


    vocab.txt:   0%|          | 0.00/996k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/125 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/895 [00:00<?, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/711M [00:00<?, ?B/s]


### LoRA


```python
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 8,
    target_modules = ['query', 'value'],
    lora_dropout = 0.05,
    bias = 'none',
    task_type = 'SEQ_CLS'
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='none'
)
```

    c:\Users\BizSpring\Desktop\lora\.venv\lib\site-packages\transformers\training_args.py:1594: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    


    model.safetensors:   0%|          | 0.00/711M [00:00<?, ?B/s]



```python
model = get_peft_model(model, lora_config)
model = model.to(device)

trainer = Trainer(
    model = model,
    args = training_args,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    compute_metrics = compute_metrics
)
```


```python
# trainer.train() # train lora
```
