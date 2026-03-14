# Visualizing Models, Data and Training with TensorBoard

This project demonstrates how to use TensorBoard alongside PyTorch to visually monitor every stage of a deep learning workflow from inspecting the raw data, through training, to evaluating model performance. Instead of relying on print statements to track loss, we log everything to TensorBoard and get interactive, browser-based dashboards.

The dataset used here is **Fashion-MNIST**, a collection of 70,000 grayscale 28×28 images across 10 clothing categories (T-shirts, trousers, sneakers, etc.). The model is a small LeNet-style CNN trained to classify these images.

---

## Project Structure

```
.
├── README.md                        # This file
├── lab_notebook.ipynb               # Main notebook with all code and comments
├── requirements.txt                 # Python dependencies
├── assets/                          # Screenshots of the Tensorboard dashboard
├── data/                            # Fashion-MNIST dataset (auto-downloaded on first run)
└── runs/                            # TensorBoard log directory (auto-created by SummaryWriter)
    └── fashion_mnist_experiment_1/
```

---

## Dependencies

```
torch
torchvision
matplotlib
numpy
tensorboard
```

Install with `pip install -r requirements.txt` inside a Python 3.9–3.12 virtual environment. No GPU is required, the model is small enough to train on CPU in a few minutes.

---

## What the Notebook Does

### Data Loading and Preprocessing

The notebook pulls the Fashion-MNIST dataset via `torchvision.datasets.FashionMNIST`, which downloads it into a local `./data/` folder on the first run. A `Compose` transform pipeline first converts each PIL image to a tensor (pixel values in [0, 1]) and then normalizes it to the range [-1, 1] using `Normalize((0.5,), (0.5,))`. The single-element tuples reflect the fact that these are single-channel grayscale images. Both the training and test splits are wrapped in `DataLoader` objects with a batch size of 4. The training loader shuffles its data each epoch; the test loader does not.

### Model Architecture

The CNN follows a classic LeNet-inspired design adapted for 28×28 grayscale input. The first convolutional layer maps 1 input channel to 6 feature maps using a 5×5 kernel, followed by ReLU and 2×2 max pooling (output: 6×12×12). The second convolutional layer expands to 16 feature maps with another 5×5 kernel, again followed by ReLU and pooling (output: 16×4×4). The resulting 256-dimensional feature vector is then passed through three fully connected layers (256→120→84→10), with ReLU activations between them. The final layer outputs raw logits for the 10 classes, there is no softmax here because `CrossEntropyLoss` handles that internally.

### Loss Function and Optimizer

The loss function is `nn.CrossEntropyLoss`, which combines log-softmax and negative log-likelihood loss in a single step. The optimizer is standard SGD with a learning rate of 0.001 and momentum of 0.9.

### TensorBoard Writer Initialization

A `SummaryWriter` is created pointing to `runs/fashion_mnist_experiment_1`. This is the central object that all TensorBoard logging goes through. Every `add_*` call serializes event data into this directoryand TensorBoard reads from it when the UI is launched. The directory is created automatically the moment the writer is instantiated.

### Logging a Sample Image Grid

Before training even starts, the notebook grabs one batch of 4 random training images and arranges them side-by-side using `torchvision.utils.make_grid`. This grid is displayed inline with matplotlib and simultaneously written to TensorBoard via `writer.add_image`. This serves as a quick sanity check — you can visually confirm that the data loaded correctly and the normalization looks reasonable.

### Logging the Model Graph

`writer.add_graph(net, images)` traces a forward pass of one batch through the network and records the full computational graph. In TensorBoard's **Graphs** tab, you can see every operation the model performs — convolutions, pooling, reshaping, linear layers — and double-click into submodules to expand them. This is especially useful when working with complex architectures where the flow of data is not immediately obvious from the code alone.

### Embedding Projector

To understand how the model "sees" the data, 100 random training images are selected and each 28×28 image is flattened into a 784-dimensional vector. These vectors are logged via `writer.add_embedding` along with their class labels and thumbnail images. TensorBoard's **Projector** tab then runs dimensionality reduction (PCA or t-SNE) interactively, letting you rotate, zoomand explore how different clothing categories cluster in the high-dimensional feature space. You can visually confirm, for instance, that sneakers and ankle boots sit near each other while trousers are far from bags.

### Training Loop with Live Logging

The model trains for 1 epoch over approximately 15,000 mini-batches. Every 1,000 batches, two things are logged:

1. **Training loss** — the average loss over the last 1,000 batches is recorded via `writer.add_scalar`. This shows up as a line chart in the **Scalars** tab, giving a smooth view of how quickly the model is converging.

2. **Prediction figures** — a matplotlib figure is generated showing the 4 images in the current batch alongside the model's predicted class, its confidence percentageand the true label. Correct predictions are colored green, incorrect ones red. These figures are logged with `writer.add_figure` and appear in the **Images** tab. Scrolling through them chronologically lets you watch the model go from random guessing to confident, correct classifications.

Two helper functions support this. `images_to_probs` runs a forward pass and extracts the predicted class index along with its softmax probability for each image in a batch. `plot_classes_preds` uses that output to build the annotated matplotlib figure described above.

### Test Set Evaluation and Precision-Recall Curves

After training completes, the notebook runs inference over the entire 10,000-image test set with gradients disabled (`torch.no_grad()`). For every test image, the softmax probability distribution across all 10 classes is collected. These probabilities and the true labels are then used to generate a precision-recall curve for each class via `writer.add_pr_curve`.

The resulting curves appear in TensorBoard's **PR Curves** tab. Classes the model handles well (like trousers, which are visually distinct) show area under the curve near 1.0, while harder classes (like shirts vs. T-shirts, which look similar) show lower values. This gives a much more nuanced picture of model performance than a single accuracy number.

---

## TensorBoard Tabs Summary

| Tab | What It Shows | How It Gets There |
|---|---|---|
| **Images** | Sample image grids and per-batch prediction figures | `add_image`, `add_figure` |
| **Scalars** | Training loss plotted over mini-batch steps | `add_scalar` |
| **Graphs** | Full computational graph of the CNN | `add_graph` |
| **Projector** | Interactive 2-D / 3-D embedding visualization of image features | `add_embedding` |
| **PR Curves** | Per-class precision-recall curves on the test set | `add_pr_curve` |

---

## How to Launch TensorBoard

From the terminal, run:

```bash
tensorboard --logdir=runs
```

Then open [http://localhost:6006](http://localhost:6006). Alternatively, inside a Jupyter notebook:

```python
%load_ext tensorboard
%tensorboard --logdir=runs
```

---
