# Neural Style Transfer using VGG19 (PyTorch)

## Overview

This project implements Neural Style Transfer (NST) using a pretrained VGG19 convolutional neural network in PyTorch. The goal is to generate a new image that combines the content of one image with the artistic style of another.

The implementation follows the original NST approach by optimizing a target image to minimize both content and style losses.

---

## Features

* Uses pretrained VGG19 from torchvision
* Extracts multi-level features for style representation
* Implements Gram matrix for texture learning
* Supports GPU acceleration (CUDA)
* Saves and displays stylized output

---

## Project Structure

```
├── content.png
├── style.png
├── stylized_output.jpeg
├── style_transfer.py
└── README.md
```

---

## Installation

### Requirements

* Python 3.x
* PyTorch
* torchvision
* Pillow
* matplotlib

### Install dependencies

```bash
pip install torch torchvision pillow matplotlib
```

---

## How It Works

### 1. Image Preprocessing

Images are resized, converted into tensors, and normalized using ImageNet statistics to match VGG19 input requirements.

### 2. Feature Extraction

A pretrained VGG19 model is used to extract features from selected layers:

* conv1_1
* conv2_1
* conv3_1
* conv4_1
* conv4_2 (content layer)
* conv5_1

### 3. Content Loss

Content loss ensures the generated image maintains the structure of the content image.

[
Content\ Loss = (F_{target} - F_{content})^2
]

### 4. Style Loss

Style is captured using Gram matrices, which represent correlations between feature maps.

[
Gram = F \cdot F^T
]

[
Style\ Loss = (G_{target} - G_{style})^2
]

### 5. Total Loss

The final loss is a weighted combination of content and style losses:

[
Total\ Loss = \alpha \cdot Content\ Loss + \beta \cdot Style\ Loss
]

Default weights:

* Content weight = 1e4
* Style weight = 1e-3

### 6. Optimization

The target image is initialized from the content image and iteratively updated using the Adam optimizer.

---

## Usage

1. Place your content and style images in the project directory.
2. Update the paths in the script:

```python
content_path = "content.png"
style_path = "style.png"
```

3. Run the script:

```bash
python style_transfer.py
```

4. Output will be saved as:

```
stylized_output.jpeg
```

---

## Output

The generated image preserves:

* Content: structure and objects from the content image
* Style: textures, colors, and patterns from the style image

---

## Customization

You can tune the following parameters:

### Change Style Strength

```python
style_weight = 1e-3
```

### Change Content Importance

```python
content_weight = 1e4
```

### Increase Quality

```python
epochs = 1000
```

### Adjust Image Size

```python
max_size = 512
```

---

## Key Concepts

* Convolutional Neural Networks capture hierarchical image features
* Content is represented by deeper layers
* Style is represented by feature correlations (Gram matrices)
* Optimization updates pixels directly instead of training the model

---

## Limitations

* Slow for high-resolution images
* Requires careful tuning of weights
* May produce artifacts if not properly balanced

---

## Future Improvements

* Fast Neural Style Transfer using feedforward networks
* Real-time style transfer using GANs
* Multi-style blending
* Video style transfer

---

## References

* Gatys et al., "A Neural Algorithm of Artistic Style"
* PyTorch Documentation
* torchvision models

---

## License

This project is for educational purposes and can be modified or extended for research and development.
