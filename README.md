# DeepLense Evaluation Tasks

This repository contains my solutions for the ML4SCI DeepLense evaluation tests. The repository contains notebooks for standard deep learning, Physics-Guided Neural Networks (PINNs) for gravitational lensing analysis and Physics-Informed Diffusion Model for Gravitational Lensing Simulation.

### Overview of Notebooks
1. `deeplense-multiclass-classification.ipynb` (Common Task I): Multi-class classification of strong lensing images using standard CNN baselines.
2. `auxiliary-pinn-classification.ipynb` (Specific Task VII): Implementation of an Auxiliary PINN (AuxPINN) that embeds physical constraints via the `caustics` simulator.
3. `evaluation-on-real-galaxies.ipynb` (independent-experiment): A domain-shift evaluation testing of ResNet34 and AuxPINN for the sim-to-real gap by overlaying simulated substructures onto real DECals galaxy backgrounds.
4. `diffusion.ipynb` (Specific Task VIII): Implementation of a custom NanoDiT (Diffusion Transformer) to generate lensing images with explicit mass conservation during inference. (For proposal-2: Physics-Informed Diffusion Models for Gravitational Lensing Simulation)

---

## Common Test I: Multi-Class Classification

**Dataset and Preprocessing**
* **Classes:** No Substructure, Sphere Substructure, Vortex Substructure.
* **Preprocessing:** Standardized image normalization, random horizontal flipping, and random rotations to improve generalization. 

**Model Architecture**
* Implemented a **ResNet-18** baseline initialized with pretrained ImageNet weights. 
* Modified the final fully connected layer to output the 3 target classes.
* Trained using Cross-Entropy Loss and the Adam optimizer.

**Results**
* **Macro AUC:** 0.9804
* **Test Accuracy:** 90.67%
* **Test Loss:** 0.2547

---

## Specific Test VII: Physics-Guided ML

**Approach for Integrating Physics Constraints**
To prevent the model from relying purely on statistical pattern matching, I implemented a dual-head Aux-PINN using the PyTorch `caustics` simulator. 

1. **Encoder for Feature Extraction:** A ResNet-34 backbone extracts a 512-dimensional feature vector from the input image.
2. **Auxiliary Physics Head:** A secondary linear layer predicts the Einstein Radius. This is bounded using a scaled Sigmoid function to ensure physically realistic outputs. 
3. **Classification Component:** The predicted physical parameter is concatenated back into the 512-dimensional feature vector. This combined 513-dimensional tensor is then passed to the final classification head.

*(Note: The model was optimized using PyTorch 2.0's `torch.compile(mode="reduce-overhead")` for faster training).*

**Results**
* **Macro-averaged ROC AUC:** 0.9925
* **Test Accuracy:** 95.28%
* **Test Loss:** 0.1514

---

### Domain Shift Stress Test (Real Galaxy Backgrounds)

**Objective**
To evaluate the true robustness of the models, I created a test set by forward-modeling the simulated dark matter substructures onto real background galaxies extracted from the DECals survey `.h5` files.

**Findings & Failure Modes:**
When subjected to complex galaxy morphologies and real telescope noise, both models failed to generalize, highlighting the core sim-to-real gap:
* **Baseline ResNet34:** Suffered complete mode collapse (predicted 'vortex' for ~100% of samples).
* **AuxPINN:** Accuracy degraded to **33.33%** (random guessing).

**Conclusion**
The stress test proves that the physical residual loss overpowered the primary classification gradients when exposed to noise (Gradient Domination). Solving this bottleneck by exploring techniques like dynamic loss weighting, and domain adaptation, etc. forms the basis of my proposed GSoC project.

## Specific Test VIII: Diffusion Models (For Physics-Informed Diffusion Models project)

**Objective:** To synthesize strong gravitational lensing images using a continuous-time generative model, proving that physical conservation laws can be mathematically enforced during the deep learning generation process.

**Approach & Architecture:**
* **Dataset:** 10,000 simulated lensing images resized to 64x64 and normalized to [-1, 1].
* **Backbone:** Built a custom **NanoDiT (Diffusion Transformer)**. The images were divided into 4x4 patches and processed through 6 transformer blocks (hidden dim: 256, heads: 8) using adaptive Layer Normalization (adaLN) for timestep conditioning. 
* **Training:** Optimized using AdamW (`lr=1e-4`) over 200 epochs on a multi-GPU setup to learn the continuous vector field.

**Physics-Constrained Inference:**
Standard unconstrained models treat generation as a statistical task, which causes them to hallucinate mass. To fix this, I implemented a custom **Physics-Constrained Sampling** loop:
* Used a 100-step Euler ODE solver for inference.
* At every integration step, the physical residual (generated mass vs. expected mass prior) is calculated.
* The latent trajectory is mathematically projected (`z_proj = z - 0.05 * residual`) to dynamically force the pixels to conserve mass before predicting the next step.

**Results & Analysis:**

Best Validation MSE loss: 0.01795

| Inference Method | FID ↓ | Mass Error (Physical Residual) ↓ |
| :--- | :--- | :--- |
| Standard DiT Baseline (Unconstrained) | **25.6673** | 0.013174 |
| Physics-Constrained Sampling | 42.0216 | **0.004389** |

**Conclusion:**
By enforcing astrophysical constraints during the Euler integration, the physical mass error was **reduced by ~66.7%**. This highlights the physics gap in standard generative deep learning, forming the foundational motivation for my second GSoC proposal for: Physics-Informed Diffusion Models.
