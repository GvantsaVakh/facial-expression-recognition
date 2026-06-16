## Experiments

This project uses the Facial Expression Recognition dataset, where each input image is a 48×48 grayscale face image and the target is one of seven emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The experiments were developed step by step. I started with a simple CNN baseline to create a working training pipeline, then improved the architecture by adding more convolutional capacity and regularization.

---

# Experiment 01: Simple CNN Baseline

## Goal

The goal of the first experiment was to build a simple baseline model for facial expression recognition. I wanted to start with a small architecture before adding more advanced techniques. This made it easier to check whether the dataset loading, preprocessing, training loop, validation loop, and WandB logging were working correctly.

A simple CNN is a natural baseline for this task because the input is image data. Convolutional layers can learn local visual patterns such as edges, eyes, mouth shapes, eyebrows, and facial contours. These patterns are important for recognizing facial expressions.

## Architecture

The baseline model used two convolutional layers followed by fully connected layers.

```text
Input: 1 × 48 × 48 grayscale image

Conv2d(1 → 16, kernel_size=3, padding=1)
ReLU
MaxPool2d(2)          # 48 × 48 → 24 × 24

Conv2d(16 → 32, kernel_size=3, padding=1)
ReLU
MaxPool2d(2)          # 24 × 24 → 12 × 12

Flatten               # 32 × 12 × 12 = 4608 features

Linear(4608 → 128)
ReLU

Linear(128 → 7)
```

This model was intentionally simple. It did not use Batch Normalization, Dropout, data augmentation, or class balancing. Because of that, it was useful as a clean baseline.

## Why I Chose This Architecture

I chose this architecture because it is small, fast to train, and easy to interpret. Since this was the first experiment, the purpose was not to build the best possible model immediately. The purpose was to create a reliable baseline that later experiments could be compared against.

The model has enough capacity to learn basic facial features, but it is still simple enough to show whether additional improvements are actually useful.

## Parameter Tuning

I tested the baseline with different training settings. The first main run used:

```text
learning rate = 0.001
epochs = 10
batch size = 64
optimizer = Adam
loss = CrossEntropyLoss
```

This run showed that the model was learning, but validation accuracy was still limited.

Then I modified the training setup by lowering the learning rate and increasing the number of epochs:

```text
learning rate = 0.0007
epochs = 15
batch size = 64
optimizer = Adam
loss = CrossEntropyLoss
```

This worked better. The smaller learning rate made training more controlled, and the larger number of epochs allowed the model to continue improving.

## Results

The best baseline run achieved approximately:

```text
Best validation accuracy: 50.02%
```

The training accuracy continued to increase during later epochs, but validation accuracy improved more slowly. This showed that the model was learning the training data, but it also started to overfit.

The baseline result was useful because it gave a clear comparison point for later experiments.

## Baseline Observation

The SimpleCNN baseline proved that the training pipeline was correct and that a CNN could learn useful features from the FER images. However, the model was limited by its small capacity and lack of regularization.

The main problems were:

```text
- validation accuracy stopped improving strongly after some epochs
- training accuracy became much higher than validation accuracy
- validation loss started increasing in later epochs
```

This suggested that the next experiment should use a stronger CNN architecture and regularization methods.

---

# Experiment 02: CNN with BatchNorm, Dropout, and Three Convolutional Blocks

## Goal

The goal of the second experiment was to improve the SimpleCNN baseline while still keeping the model small enough to train on Google Colab GPU.

The first model showed signs of overfitting and had limited feature extraction capacity. Therefore, in the second experiment, I added:

```text
- one additional convolutional block
- more convolutional channels
- Batch Normalization
- Dropout
- weight decay
- learning rate scheduling
```

This experiment was designed as a controlled improvement over the baseline, not as a completely different model.

## Final Architecture

The final version of Experiment 02 used three convolutional blocks followed by a flatten-based classifier.

```text
Input: 1 × 48 × 48 grayscale image

Conv2d(1 → 32, kernel_size=3, padding=1)
BatchNorm2d(32)
ReLU
MaxPool2d(2)          # 48 × 48 → 24 × 24

Conv2d(32 → 64, kernel_size=3, padding=1)
BatchNorm2d(64)
ReLU
MaxPool2d(2)          # 24 × 24 → 12 × 12

Conv2d(64 → 128, kernel_size=3, padding=1)
BatchNorm2d(128)
ReLU
MaxPool2d(2)          # 12 × 12 → 6 × 6

Flatten               # 128 × 6 × 6 = 4608 features

Dropout(0.3)

Linear(4608 → 256)
ReLU

Dropout(0.3)

Linear(256 → 7)
```

## Why I Chose This Architecture

This model is a natural next step after the SimpleCNN baseline.

The baseline used two convolutional layers with 16 and 32 output channels. In Experiment 02, I increased the number of channels to 32, 64, and 128. This gives the model more capacity to learn facial features.

I also added a third convolutional block. The first layers can learn simple features such as edges and small textures, while deeper layers can combine them into more meaningful facial patterns, such as eyes, mouth shapes, and eyebrow positions.

Batch Normalization was added after every convolutional layer to make training more stable. Dropout was added before the fully connected layers to reduce overfitting. Weight decay was also used as another regularization method.

## Stabilization Problem

Before reaching the final version of Experiment 02, I tried a different architecture that used Global Average Pooling.

That version looked like this:

```text
Conv blocks
AdaptiveAvgPool2d(1)
Flatten
Linear(128 → 7)
```

The idea was to reduce the number of parameters and make the classifier smaller. However, this version performed worse and validation accuracy was unstable.

The likely reason was that `AdaptiveAvgPool2d(1)` compressed each feature map into a single value. After this operation, the classifier received only 128 features. For facial expression recognition, this was probably too aggressive because expressions depend on spatial details: the location of the eyes, eyebrows, mouth, and other facial regions matters.

In other words, the Global Average Pooling version removed too much spatial information before classification.

## How I Fixed the Problem

To fix this, I removed Global Average Pooling and used a flatten-based classifier instead.

Instead of giving the classifier only 128 features, the corrected model keeps the final 6×6 spatial feature map:

```text
128 × 6 × 6 = 4608 features
```

This gives the classifier much more information about where facial features appear in the image.

The corrected model still uses Batch Normalization, Dropout, and weight decay, so it remains regularized, but it no longer compresses the spatial information too early.

## Hyperparameters

The final Experiment 02 setup used approximately:

```text
learning rate = 0.0005
epochs = 15
batch size = 64
optimizer = Adam
loss = CrossEntropyLoss
dropout = 0.3
weight decay = 1e-4
scheduler = ReduceLROnPlateau
```

I used a slightly lower learning rate than the baseline because the model is deeper and has more parameters. A lower learning rate helped make training more stable.

## Results

The corrected Experiment 02 model performed better than the SimpleCNN baseline.

Approximate validation results:

```text
Experiment 01 SimpleCNN baseline:     best val accuracy ≈ 50.02%
Experiment 02 improved CNN:           best val accuracy ≈ 56%
```

This is an improvement of about 6 percentage points over the baseline.

The validation loss also generally decreased, and validation accuracy improved across training. This showed that the final version of the second architecture generalized better than the first baseline and better than the failed Global Average Pooling version.

## Experiment 02 Observation

Experiment 02 showed that increasing convolutional capacity helped the model learn better facial features. Adding Batch Normalization, Dropout, and weight decay made the deeper model more trainable and controlled overfitting.

The most important lesson from this experiment was that architecture design is not only about adding more layers. The way the final feature representation is passed to the classifier also matters. Global Average Pooling reduced the representation too much for this task, while the flatten-based classifier preserved more spatial information and produced better validation accuracy.

---

# Summary of First Two Experiments

| Experiment    | Main Change                                        | Best Validation Accuracy | Observation                                                   |
| ------------- | -------------------------------------------------- | -----------------------: | ------------------------------------------------------------- |
| Experiment 01 | SimpleCNN baseline                                 |                  ~50.02% | Good starting point, but limited capacity and overfitting     |
| Experiment 02 | 3 Conv blocks + BatchNorm + Dropout + weight decay |                     ~56% | Better feature extraction and stronger validation performance |

The first two experiments show a clear development path. The baseline created a simple working CNN pipeline. The second experiment improved the architecture by increasing convolutional capacity and adding regularization. The corrected second model became a stronger base for later experiments such as data augmentation, class imbalance handling, or residual architectures.
