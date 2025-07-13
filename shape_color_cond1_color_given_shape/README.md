# Conditional Model: P(color | shape) × P(shape)

This project implements the **conditional generative model** for colored shapes based on the compositional generative modeling paper: _"A Single Model is Not All You Need"_.

## 📌 Goal
Learn to color grayscale shapes using a model that approximates:

```
P(shape, color) = P(color | shape) × P(shape)
```

- **P(shape)** is implicitly learned by the grayscale shape VAE.
- **P(color | shape)** is learned by a CNN that maps grayscale shapes to RGB versions.

---

## 📂 Folder Structure
```
.
├── scripts/
│   ├── train_color_mapper.py         # Training script
│   └── view_results.ipynb           # Visualize input/output pairs
├── color_mapper.py                  # U-Net-like model (1-channel → 3-channel)
├── color_dataset.py                 # Dataset for grayscale input + RGB targets
├── samples/cond1/                   # Saved outputs every N epochs
└── models/                          # Trained model weights
```

---

## 🧠 Model Architecture
### `ColorMapper`
- Input: 1×64×64 grayscale image
- Output: 3×64×64 color prediction
- Structure: 2 conv layers → middle conv → 2 transposed conv layers

### Loss Function
```
loss = MSE(pred, target) + 
       edge_weight × EdgeLoss(pred, target)
```

- `EdgeLoss` encourages the predicted color transitions to align with shape edges.

---

## 🚀 Training
```bash
python scripts/train_color_mapper.py
```
Outputs will be saved to:
```
samples/cond1/output_color_epochXX.png
samples/cond1/input_gray_epochXX.png
```

---

## 🔍 Evaluation
Use `view_results.ipynb` to compare grayscale inputs and predicted color outputs.

Unseen combinations such as **green triangle** or **blue square** indicate generalization ability.

---

## 📈 Result Highlights
- Accurate shape-to-color prediction from grayscale inputs
- Sharp color boundaries
- Generalization to unseen combinations in some cases

---

## 🔄 Next Steps
Try training alternate factorizations:
- `P(color) × P(shape | color)`
- Joint `P(shape, color)` as a single VAE

Then compare all four models under a fixed training budget.

---

## 📬 Author
Abhiram Varma Nandimandalam  
M.S. Data Science @ University of Arizona  
[GitHub](https://github.com/isjustabhi)
