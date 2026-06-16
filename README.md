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


## Experiment 03: Data Augmentation and Regularized Training

### Goal

After Experiment 02, the model had improved compared to the baseline, but I still wanted to improve generalization. The previous CNN architecture was already stronger than the baseline, so in Experiment 03 I kept the same model architecture and focused on the training strategy instead.

The main goal of this experiment was to reduce overfitting and make the model more robust to small variations in facial images.

### Architecture

Experiment 03 used the same architecture as Experiment 02:

```text
Conv(1 → 32) → BatchNorm → ReLU → MaxPool
Conv(32 → 64) → BatchNorm → ReLU → MaxPool
Conv(64 → 128) → BatchNorm → ReLU → MaxPool
Flatten
Dropout(0.3)
Linear(128 * 6 * 6 → 256)
ReLU
Dropout(0.3)
Linear(256 → 7)
```

The model had:

```text
1,274,823 trainable parameters
```

This was useful because the architecture stayed fixed, so the effect of the training changes could be analyzed more clearly.

### What I Changed

In this experiment, I added data augmentation to the training set. The augmentation included:

```text
Random horizontal flip
Small random rotation
Small random translation
```

The purpose was to make the model less dependent on exact face position and alignment. Since the images are only 48×48 pixels, I kept the augmentation relatively mild. Strong transformations could distort facial expressions and hurt performance.

The validation set was not augmented, because validation should measure performance on unchanged images.

In the logged run, I also used class weighting with `WeightedCrossEntropyLoss`. This was tested because the FER dataset is imbalanced: some emotion classes appear much more often than others. Class weighting gives more importance to rare classes during training.

### Result

Experiment 03 achieved:

```text
Best validation accuracy: 57.45%
Final validation accuracy: 56.87%
Final training accuracy: 58.60%
Final training loss: 1.0982
Final validation loss: 1.1048
```

### Analysis

The most important result of Experiment 03 was not only the validation accuracy, but the relationship between training and validation performance.

The final training accuracy was about 58.60%, while the final validation accuracy was about 56.87%. These values are close to each other, which shows that the model was no longer strongly overfitting. Compared to earlier experiments, the gap between training and validation performance became smaller.

This suggests that data augmentation acted as a regularizer. It made the training task harder because the model saw slightly modified images during training, but this helped the model generalize better.

Class weighting was also tested as a way to handle class imbalance. However, this kind of loss can sometimes reduce overall accuracy because the model gives more attention to rare classes. In this experiment, the final result was still useful because it showed that training strategy changes can reduce overfitting and improve robustness.

Overall, Experiment 03 improved the training behavior of the model. It did not dramatically increase accuracy, but it made the model more stable and less overfitted.

---

## Experiment 04: VGG-Style CNN with Stronger Feature Extraction

### Goal

After Experiment 03, the model generalized better, but the validation accuracy was still around 57%. For Experiment 04, I wanted to improve the model architecture again, but still stay within a custom CNN approach instead of immediately switching to a pretrained model.

The goal was to build a stronger CNN feature extractor while keeping the model small enough to train on Google Colab GPU.

### Architecture

Experiment 04 used a VGG-style CNN architecture. The main change was that each convolutional block now had two convolutional layers before max pooling.

The architecture was:

```text
Block 1:
Conv(1 → 32) → BatchNorm → ReLU
Conv(32 → 32) → BatchNorm → ReLU
MaxPool

Block 2:
Conv(32 → 64) → BatchNorm → ReLU
Conv(64 → 64) → BatchNorm → ReLU
MaxPool

Block 3:
Conv(64 → 128) → BatchNorm → ReLU
Conv(128 → 128) → BatchNorm → ReLU
MaxPool

Flatten
Dropout(0.35)
Linear(128 * 6 * 6 → 512)
ReLU
Dropout(0.35)
Linear(512 → 7)
```

The model had:

```text
2,650,727 trainable parameters
```

### Why I Chose This Architecture

In Experiment 02 and Experiment 03, each block had only one convolutional layer before pooling. This means the spatial size was reduced quickly.

In Experiment 04, I used two convolutional layers before each pooling operation. This allows the model to learn richer local patterns before reducing the image resolution.

This is important for facial expression recognition because expressions depend on small facial details, such as:

```text
mouth corners
eyebrow position
eye shape
cheek movement
local facial texture
```

The VGG-style design gives the model more time to extract these local features before max pooling reduces the spatial dimensions.

### Training Setup

Experiment 04 used:

```text
Optimizer: AdamW
Learning rate: 0.0005
Weight decay: 0.0005
Dropout: 0.35
Epochs: 25
Batch size: 64
Augmentation: horizontal flip + small rotation
Scheduler: learning rate reduction during training
```

I used AdamW because it handles weight decay better than standard Adam. I also increased weight decay and used Dropout to control overfitting, because this model has more parameters than the previous CNNs.

### Result

Experiment 04 achieved the best result so far:

```text
Best validation accuracy: 63.70%
Final validation accuracy: 62.98%
Final training accuracy: 72.59%
Final training loss: 0.7281
Final validation loss: 1.0089
Final learning rate: 0.000125
```

### Analysis

Experiment 04 gave a large improvement over the previous experiments.

The best validation accuracy increased from about 57.45% in Experiment 03 to 63.70% in Experiment 04. This shows that improving the convolutional feature extractor was more effective than only changing the training strategy.

The model did start to overfit more than Experiment 03, because the final training accuracy was 72.59% while validation accuracy was 62.98%. However, this gap is expected because the model is larger and more expressive. The important point is that validation accuracy also improved significantly, so the additional capacity was useful.

The learning rate scheduler also helped training. By the end of training, the learning rate had been reduced to 0.000125, which allowed the model to continue improving more carefully.

Overall, Experiment 04 became the strongest custom CNN model in the project. It showed that a deeper VGG-style CNN with stronger feature extraction can perform much better than the simpler CNN architectures.

---

## Comparison of Experiments 03 and 04

| Experiment    | Main Idea                                      | Train Accuracy | Validation Accuracy | Best Validation Accuracy | Observation                                                      |
| ------------- | ---------------------------------------------- | -------------: | ------------------: | -----------------------: | ---------------------------------------------------------------- |
| Experiment 03 | Same CNN + augmentation / regularized training |         58.60% |              56.87% |                   57.45% | Reduced overfitting and made train/validation performance closer |
| Experiment 04 | VGG-style CNN with two convolutions per block  |         72.59% |              62.98% |                   63.70% | Much stronger feature extraction and best custom CNN result      |

Experiment 03 mainly improved generalization and reduced overfitting. Experiment 04 improved the architecture itself and achieved the best validation accuracy so far.
