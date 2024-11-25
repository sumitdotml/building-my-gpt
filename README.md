# SIMPLEST GPT-2

This is me attempting to replicate GPT-2. At the moment, I've only managed to finish building a simple transformer block that can generate embeddings. I'm currently working on the rest of the model. It's not very useful at the moment, but it's a good starting point.

To start, you can create a virtual environment and install the dependencies:

```
install -r requirements.txt
```

You can change the input text in the [run.py](./run.py) file as well as configure the model parameters in the `GPT_CONFIG_124M` dictionary. Based on the input text, the model will generate embeddings.

To run the code, you can use the following command:

```
python run.py
```

---
### COMPLETED
- [x] Multi-head attention
- [x] Feed-forward network
- [x] Layer normalization
- [x] Transformer block

---

### TODO

- [ ] convert the out tensors back to text
- [ ] Pretraining
- [ ] Fine-tuning