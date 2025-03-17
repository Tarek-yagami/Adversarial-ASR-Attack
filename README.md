# Adversarial Attack on Wav2Vec2

## 📌 Description  
This project demonstrates an adversarial attack against the Wav2Vec2 ASR model.  
By adding small, imperceptible perturbations to an audio signal, we can manipulate the model's transcription while keeping the modifications inaudible to humans.  

## ⚡ Methods Implemented  
- **Projected Gradient Descent (PGD)** attack on CTC loss  
- **Psychoacoustic Masking** to hide perturbations  
- **Energy-Based Filtering** to make the attack more discreet  


## 🔧 Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Tarek-yagami/Adversarial-ASR-Attack.git
   cd Adversarial-ASR-Attack
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage  
Run the Jupyter Notebook to test the attack:  
```bash
jupyter notebook
```
Open `attack_wav2vec2.ipynb` and follow the steps.  



## 📖 References  
- **Wav2Vec2 Paper** - [https://arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)  
- **Adversarial Attacks on ASR** - [https://arxiv.org/abs/1801.01944](https://arxiv.org/abs/1801.01944)  
