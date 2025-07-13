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
<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/c0acebd8-8a99-439b-886a-bfef1aa1eceb" />
<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/c913cb72-8c63-46cd-a777-364396dc595c" />
<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/fb175a72-d101-4b56-b227-4903a14c8f14" />

---

## 📈 Result Highlights
- Accurate shape-to-color prediction from grayscale inputs
- Sharp color boundaries
- Generalization to unseen combinations in some cases

---

## 📬 Author
Abhiram Varma Nandimandalam  
M.S. Data Science @ University of Arizona  
[GitHub](https://github.com/isjustabhi)
