# DeepLense GSoC 2026 Evaluation Tasks

This repository contains my solutions for the ML4SCI DeepLense evaluation tests. The project explores both standard deep learning and Physics-Informed Neural Networks (PINNs) for gravitational lensing analysis, concluding with a stress test on real-world observational data.

### Overview of Notebooks
1. `deeplense-multiclass-classification.ipynb` (Common Task I): Multi-class classification of strong lensing images using standard CNN baselines.
2. `auxiliary-pinn-classification.ipynb` (Specific Task VII): Implementation of an Auxiliary PINN (AuxPINN) that embeds physical constraints via the `caustics` simulator.
3. `evaluation-on-real-galaxies.ipynb` (independent-experiment): A domain-shift evaluation testing of ResNet34 and AuxPINN for the sim-to-real gap by overlaying simulated substructures onto real DECals galaxy backgrounds.

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

## Specific Test VII: Physics Informed Neural Network (PINN)

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

## Domain Shift Stress Test (Real Galaxy Backgrounds)

**Objective**
To evaluate the true robustness of the models, I created a test set by forward-modeling the simulated dark matter substructures onto real background galaxies extracted from the DECals survey `.h5` files.

**Findings & Failure Modes:**
When subjected to complex galaxy morphologies and real telescope noise, both models failed to generalize, highlighting the core sim-to-real gap:
* **Baseline ResNet34:** Suffered complete mode collapse (predicted 'vortex' for ~100% of samples).
* **Aux-PINN:** Accuracy degraded to **33.33%** (random guessing). 

**Conclusion**
The stress test proves that the physical residual loss overpowered the primary classification gradients when exposed to noise (Gradient Domination). Solving this bottleneck via dynamic loss weighting and domain adaptation forms the basis of my proposed GSoC project.