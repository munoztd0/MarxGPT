# South Park GPT Language Model
This project trains a GPT (Generative Pre-trained Transformer) language model on the South Park corpus. The model can generate new text based on a given context.

# Requirements
Python 3.6 or higher
PyTorch 1.7 or higher
CUDA-enabled GPU (optional, but recommended)
Installation


# Install the required packages:

```python
pip install -r requirements.txt

```



To generate new text from the model, run the following command:

```python
python gpt.py
```


# Configuration
You can configure the hyperparameters of the model by modifying the variables in the gpt.py file. Here are the default values:

batch_size: 64
block_size: 156
max_iters: 1000
eval_interval: 500
learning_rate: 3e-4
device: 'cuda' if available, else 'cpu'
eval_iters: 100
n_embd: 284
n_head: 6
n_layer: 6
dropout: 0.2


# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
This project is based on the PyTorch GPT tutorial. The South Park corpus was obtained from Kaggle.

# Example of render

Cartman:  HellO?
 
People of here, what?!
 
Kyle: Mittle your team is nastyyy.
 
Randy: [it it howly]
 
May: [stills of to of Aman O apezus.
 
Kyle: Sorro走, son, came.
 
 
Randy: A right a get and?
 
Deto Simplestic 4: [steps a give."] And watell eveated feriod. No. Look you'reing this or need looking ney're deactiff and firound j


