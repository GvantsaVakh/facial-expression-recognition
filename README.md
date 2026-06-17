# Facial Expression Recognition Challenge

## პროექტის ბმულები

* **Weights & Biases Project:**
  https://wandb.ai/gvakh23-tbilisi-free-university/Facial_Expression_Recognition?nw=nwusergvakh23

* **Weights & Biases Report:**
  https://wandb.ai/gvakh23-tbilisi-free-university/Facial_Expression_Recognition/reports/Facial-Expression-Recognition-Experiment-Report--VmlldzoxNzI1Nzc5MA/edit?draftId=VmlldzoxNzI1Nzc5MA%3D%3D

---

## პროექტის აღწერა

ეს პროექტი შესრულებულია Kaggle-ის competition-ისთვის: **Challenges in Representation Learning: Facial Expression Recognition Challenge**.

ამოცანის მიზანია სახის სურათიდან ემოციის კლასიფიკაცია. თითოეული input არის 48×48 ზომის grayscale სახის სურათი, ხოლო target არის ერთ-ერთი 7 ემოციის კლასი:

```text
0 — Angry
1 — Disgust
2 — Fear
3 — Happy
4 — Sad
5 — Surprise
6 — Neutral
```

ამიტომ პროექტი ავაგე ეტაპობრივად. დავიწყე პატარა CNN baseline-ით და შემდეგ ნელ-ნელა გავაუმჯობესე მოდელი: დავამატე Batch Normalization, Dropout, data augmentation, VGG-style CNN არქიტექტურა და ბოლოს ResNet18 transfer learning.

---

## რეპოზიტორიის სტრუქტურა

რეპოზიტორიაში თითოეული მთავარი ექსპერიმენტი ცალკე notebook-ად არის წარმოდგენილი:

```text
.
├── README.md
├── Facial_Expression_Recognition_01.ipynb
├── Facial_Expression_Recognition_02.ipynb
├── Facial_Expression_Recognition_03.ipynb
├── Facial_Expression_Recognition_04.ipynb
├── Facial_Expression_Recognition_05.ipynb
```

notebook-ების დანიშნულება:

```text
Facial_Expression_Recognition_01.ipynb — Simple CNN baseline
Facial_Expression_Recognition_02.ipynb — Improved CNN with BatchNorm and Dropout
Facial_Expression_Recognition_03.ipynb — Data augmentation and regularized CNN
Facial_Expression_Recognition_04.ipynb — VGG-style CNN
Facial_Expression_Recognition_05.ipynb — ResNet18 transfer learning
```

მოდელების საუკეთესო checkpoint-ები შენახულია Google Drive-ზე, ხოლო საბოლოო ტესტირებისას notebook-ები checkpoint-ებს Drive-დან ტვირთავს და internal test set-ზე აფასებს.

---

მოდელების training და evaluation ძირითადად Google Colab-ზე შესრულდა. ექსპერიმენტების metrics, hyperparameters, sanity checks, confusion matrices და classification reports დალოგილია WandB-ზე.

---

## მონაცემები და preprocessing

dataset-ში სურათები ინახება pixel string-ის სახით. თითოეული row შეიცავს:

```text
emotion — target label
pixels — 48×48 grayscale image-ის pixel values string ფორმატში
```

პირველ ეტაპზე pixel string გადავაქციე NumPy array-დ:

```text
pixels string → NumPy array → reshape(48, 48)
```

შემდეგ სურათები PyTorch Dataset/DataLoader-ებში გადავიტანე. მოდელების მიხედვით preprocessing ოდნავ განსხვავდებოდა. custom CNN მოდელებისთვის input ფორმა იყო:

```text
[batch_size, 1, 48, 48]
```

ანუ 1-channel grayscale image.

ResNet18 ექსპერიმენტშიც საბოლოოდ შევინარჩუნე 48×48 grayscale input, მაგრამ ResNet-ის პირველი convolution შევცვალე ისე, რომ მიეღო 1-channel input.

---

## Train / Validation / Test split

მონაცემები გავყავი სამ ნაწილად:

```text
Training set — მოდელის სასწავლად
Validation set — hyperparameter tuning-ისთვის და best checkpoint-ის ასარჩევად
Internal test set — საბოლოო შეფასებისთვის
```

გამოვიყენე stratified split, რომ ყველა emotion class წარმოდგენილი ყოფილიყო train, validation და test split-ებში.

test set არ გამომიყენებია მოდელის ასარჩევად. თითოეულ ექსპერიმენტში საუკეთესო checkpoint შევარჩიე validation accuracy-ის მიხედვით, ხოლო ბოლოს checkpoint ჩავტვირთე Drive-დან და შევაფასე საბოლოოდ internal test set-ზე.

---

## WandB Tracking

ყველა ექსპერიმენტი დალოგილია Weights & Biases-ზე. თითოეულ არქიტექტურას აქვს ცალკე run. დალოგილი მონაცემები:

```text
training loss
validation loss
training accuracy
validation accuracy
best validation accuracy
learning rate
hyperparameters
model architecture
number of trainable parameters
final test loss
final test accuracy
forward sanity check
backward sanity check
confusion matrix
classification report
```

## Sanity checks

დავამატე sanity checks:

### Forward pass check

Forward check ამოწმებს, რომ model იღებს სწორ input shape-ს და აბრუნებს 7 logits-ს, რადგან გვაქვს 7 ემოციის კლასი.

მაგალითად output უნდა იყოს:

```text
[batch_size, 7]
```

ასევე მოწმდება, რომ output-ში არ არის NaN ან Inf მნიშვნელობები.

### Backward pass check

Backward check ითვლის loss-ს და იძახებს:

```python
loss.backward()
```

ამის შემდეგ მოწმდება, რომ პარამეტრებზე gradients ნამდვილად წარმოიქმნა და gradient norm არ არის 0.

### One-batch overfitting check

ResNet ექსპერიმენტისთვის დამატებით გავაკეთე one-batch overfitting check. მოდელი რამდენჯერმე ვატრეინინგე ერთ mini-batch-ზე. მიზანი არ იყო generalization-ის გაზომვა, არამედ იმის დამტკიცება, რომ მოდელს სწავლა შეუძლია და optimizer/loss/backward update სწორად მუშაობს.

one-batch check-ის შედეგი:

```text
Step 20/80 | Loss: 0.5913 | Accuracy: 0.9219
Step 40/80 | Loss: 0.2479 | Accuracy: 0.9688
Step 60/80 | Loss: 0.1448 | Accuracy: 1.0000
Step 80/80 | Loss: 0.0802 | Accuracy: 1.0000
```

ეს ნიშნავს, რომ მოდელმა შეძლო ერთი batch-ის სრულად დამახსოვრება. შესაბამისად training pipeline სწორად მუშაობს.

---

# Experiments

## Experiment 01 — Simple CNN Baseline

### მიზანი

პირველი ექსპერიმენტის მიზანი იყო მარტივი baseline-ის შექმნა.

```text
data loading
preprocessing
Dataset/DataLoader
training loop
validation loop
checkpoint saving
WandB logging
```

### არქიტექტურა

baseline მოდელი იყო პატარა CNN:

```text
Input: 1 × 48 × 48 grayscale image

Conv2d(1 → 16, kernel_size=3, padding=1)
ReLU
MaxPool2d(2)

Conv2d(16 → 32, kernel_size=3, padding=1)
ReLU
MaxPool2d(2)

Flatten

Linear(32 * 12 * 12 → 128)
ReLU

Linear(128 → 7)
```

### Hyperparameters

საწყისად გამოვცადე:

```text
learning rate = 0.001
epochs = 10
batch size = 64
optimizer = Adam
loss = CrossEntropyLoss
```

შემდეგ learning rate შევამცირე და epochs გავზარდე:

```text
learning rate = 0.0007
epochs = 15
batch size = 64
optimizer = Adam
loss = CrossEntropyLoss
```

ეს უკეთესი აღმოჩნდა, რადგან პატარა learning rate უფრო კონტროლირებად training-ს იძლეოდა.

### შედეგი

baseline-ის საუკეთესო validation accuracy იყო დაახლოებით:

```text
Best validation accuracy ≈ 50.02%
```

### ანალიზი

Simple CNN-მა აჩვენა, რომ pipeline სწორად მუშაობდა და მოდელს შეეძლო სახის სურათებიდან რაღაც ნიშნების სწავლა. თუმცა მოდელი პატარა იყო და ჰქონდა შეზღუდული capacity.

training accuracy იზრდებოდა, მაგრამ validation accuracy უფრო ნელა უმჯობესდებოდა. ეს მიუთითებდა overfitting-ის ნიშნებზე და იმაზე, რომ შემდეგ ექსპერიმენტში საჭირო იყო უფრო ძლიერი feature extractor და regularization.

---

## Experiment 02 — CNN with BatchNorm and Dropout

### მიზანი

მეორე ექსპერიმენტში baseline CNN გავაუმჯობესე. დავამატე:

```text
მესამე convolutional block
უფრო მეტი channels
Batch Normalization
Dropout
weight decay
learning rate scheduler
```

მიზანი იყო უკეთესი feature extraction და უფრო სტაბილური training.

### საბოლოო არქიტექტურა

```text
Input: 1 × 48 × 48 grayscale image

Conv2d(1 → 32)
BatchNorm2d(32)
ReLU
MaxPool

Conv2d(32 → 64)
BatchNorm2d(64)
ReLU
MaxPool

Conv2d(64 → 128)
BatchNorm2d(128)
ReLU
MaxPool

Flatten

Dropout(0.3)
Linear(128 * 6 * 6 → 256)
ReLU
Dropout(0.3)
Linear(256 → 7)
```

### რატომ BatchNorm და Dropout?

Batch Normalization დავამატე, რომ training უფრო სტაბილური ყოფილიყო. უფრო ღრმა CNN-ში activation distributions შეიძლება იცვლებოდეს, რაც training-ს ართულებს. BatchNorm ამ პროცესს ასტაბილურებს.

Dropout დავამატე overfitting-ის შესამცირებლად. რადგან მოდელი baseline-ზე უფრო დიდი გახდა, საჭირო იყო რეგულარიზაცია.

### პრობლემა: GAP ვერსია

ამ ექსპერიმენტში თავიდან ვცადე Global Average Pooling ვერსია:

```text
Conv blocks
AdaptiveAvgPool2d(1)
Flatten
Linear(128 → 7)
```

იდეა იყო parameters-ის შემცირება და overfitting-ის შემცირება. მაგრამ შედეგი არასტაბილური და დაბალი იყო.

GAP თითოეულ feature map-ს ერთ რიცხვად ასაშუალოებს. ანუ საბოლოო feature-ების რეპრეზენტაცია ხდება მხოლოდ:

```text
128 features
```

ამის გამო spatial information იკარგებოდა. Facial expression recognition-ში კი მნიშვნელოვანია, სად არის კონკრეტული feature მაგალითად პირი, წარბები, თვალები და მათი მდებარეობა.

### როგორ გამოვასწორე

GAP ამოვიღე და დავაბრუნე Flatten + fully connected classifier დავტოვე:

```text
128 × 6 × 6 = 4608 features
```

ამით classifier-მა მიიღო ბევრად მეტი ინფორმაცია. ამის შემდეგ validation accuracy გაუმჯობესდა და training უფრო სტაბილური გახდა.

### Hyperparameters

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

### შედეგი

```text
Experiment 01 best val accuracy ≈ 50.02%
Experiment 02 best val accuracy ≈ 56%
```

### ანალიზი

Experiment 02-მ აჩვენა, რომ convolutional capacity-ის გაზრდა და regularization რეალურად ეხმარება მოდელს. BatchNorm-მა training დაასტაბილურა, Dropout-მა და weight decay-მ კი overfitting შეამცირა.

---

## Experiment 03 — Data Augmentation and Regularized Training

### მიზანი

მესამე ექსპერიმენტში არქიტექტურა ძირითადად იგივე დავტოვე, რაც Experiment 02-ში, მაგრამ training strategy შევცვალე. მიზანი იყო overfitting-ის შემცირება და generalization-ის გაუმჯობესება.

### არქიტექტურა

Experiment 03 იყენებს იგივე CNN-ს:

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

მოდელს აქვს:

```text
1,274,823 trainable parameters
```

### რა შევცვალე

დავამატე data augmentation:

```text
Random horizontal flip
Small random rotation
Small random translation
```

სახის სურათები შეიძლება ოდნავ განსხვავებულად იყოს მდებარეობით ან კუთხით, ამიტომ augmentation ეხმარება მოდელს, რომ კონკრეტულ pixel positions-ზე ნაკლებად იყოს დამოკიდებული.

სურათები მხოლოდ 48×48 იყო, ამიტომ augmentation არ უნდა ყოფილიყო ძალიან ძლიერი. ძლიერი rotation/translation ან blur სახის პატარა დეტალებს გააფუჭებდა.

### Class imbalance

ასევე გამოვცადე class weighting / WeightedCrossEntropyLoss, რადგან FER dataset-ში კლასები არ არის თანაბრად გადანაწილებული. მაგალითად ზოგი ემოცია ბევრად იშვიათია.

class weighting-ის იდეაა იშვიათ მეტი მნიშვნელობა მიენიჭოს ტრენინგის დროს. თუმცა პრაქტიკაში ეს ყოველთვის არ ზრდის overall accuracy-ს, რადგან მოდელი შეიძლება ზედმეტად ფოკუსირდეს იშვიათ კლასებზე. ამ შემთხვევაშიც, სავარაუდოდ სწორედ ამიტომ, კარგი შედეგი არ გამოიღო დამატებამ.

### შედეგი

Experiment 03-ის შედეგი:

```text
Best validation accuracy: 57.45%
Final validation accuracy: 56.87%
Final training accuracy: 58.60%
Final training loss: 1.0982
Final validation loss: 1.1048
```

### ანალიზი

Experiment 03-ის ყველაზე მნიშვნელოვანი შედეგი ის იყო, რომ overfitting შემცირდა. training accuracy და validation accuracy ერთმანეთთან ახლოს იყო:

```text
train accuracy ≈ 58.60%
val accuracy ≈ 56.87%
```

ეს ნიშნავს, რომ მოდელი training set-ს ზედმეტად აღარ იმახსოვრებდა. augmentation-მა და regularization-მა training გაართულა, მაგრამ generalization გააუმჯობესა.

accuracy-ის ზრდა დიდი არ იყო, მაგრამ ეს ექსპერიმენტი მაინც მნიშვნელოვანი იყო, რადგან აჩვენა, როგორ მოქმედებს augmentation overfitting-ზე.

---

## Experiment 04 — VGG-style CNN

### მიზანი

მესამე არქიტექტურამ overfitting შეამცირა, მაგრამ validation accuracy დაახლოებით 57%-ის გარშემო დარჩა. ამიტომ მეოთხე ექსპერიმენტში გადავწყვიტე ისევ არქიტექტურის გაუმჯობესება.

მიზანი იყო custom CNN-ის გაძლიერება pretrained მოდელის გამოყენების გარეშე.

### VGG-style იდეა

VGG-style _ convolutional block-ში pooling-მდე ვიყენებთ რამდენიმე პატარა 3×3 convolution-ს.

წინა მოდელებში მქონდა:

```text
Conv → MaxPool
Conv → MaxPool
Conv → MaxPool
```

VGG-style მოდელში გვაქვს:

```text
Conv → Conv → MaxPool
Conv → Conv → MaxPool
Conv → Conv → MaxPool
```

### არქიტექტურა

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

მოდელს ჰქონდა:

```text
2,650,727 trainable parameters
```

როდესაც pooling ძალიან ადრე ხდება, ნაწილი spatial detail იკარგება. VGG-style CNN ჯერ ორჯერ ამუშავებს feature maps-ს convolution-ით და მხოლოდ შემდეგ ამცირებს ზომას MaxPool-ით.

### Training setup

```text
optimizer = AdamW
learning rate = 0.0005
weight decay = 0.0005
dropout = 0.35
epochs = 25
batch size = 64
augmentation = horizontal flip + small rotation
scheduler = ReduceLROnPlateau
```

AdamW გამოვიყენე, რადგან weight decay-ს უკეთესად ამუშავებს, ვიდრე ჩვეულებრივი Adam. მოდელი უფრო დიდი იყო, ამიტომ გავზარდე regularization-იც.

### შედეგი

```text
Best validation accuracy: 63.70%
Final validation accuracy: 62.98%
Final training accuracy: 72.59%
Final training loss: 0.7281
Final validation loss: 1.0089
Final learning rate: 0.000125
```

### ანალიზი

Experiment 04 იყო დიდი improvement. საუკეთესო validation accuracy გაიზარდა:

```text
Experiment 03: 57.45%
Experiment 04: 63.70%
```

ეს აჩვენებს, რომ feature extractor-ის გაძლიერება უფრო ეფექტური იყო, ვიდრე მხოლოდ training strategy-ის შეცვლა.

მოდელში overfitting ისევ გამოჩნდა, რადგან train accuracy ბევრად მაღალი იყო validation accuracy-ზე. თუმცა validation accuracy-ც მნიშვნელოვნად გაიზარდა, ამიტომ დამატებითი capacity საბოლოოდ სასარგებლო აღმოჩნდა.

Experiment 04 გახდა საუკეთესო custom CNN მოდელი.

---

## Experiment 05 — ResNet18 Transfer Learning

### მიზანი

გააუმჯობესებდა თუ არა pretrained მოდელი შედეგს custom CNN-ებთან შედარებით.

### საწყისი პრობლემა

სტანდარტული ResNet18 განკუთვნილია 224×224 RGB სურათებისთვის. FER dataset-ში კი გვაქვს:

```text
48×48 grayscale images
```

ამიტომ პირდაპირ ResNet18-ის გამოყენება იდეალური არ იყო. პირველი ცდისას ResNet ძალიან სწრაფად მიდიოდა overfit-ში: train accuracy ძალიან მაღლა ადიოდა, validation კი მალე ჩერდებოდა.

### Small-input ResNet ცვლილება

ResNet უკეთ რომ მორგებოდა 48×48 grayscale სურათებს, შევცვალე მისი პირველი ნაწილი:

```text
original ResNet conv1:
7×7 convolution, stride=2, RGB input

modified conv1:
3×3 convolution, stride=1, grayscale input
```

ასევე ამოვიღე early maxpool:

```text
resnet.maxpool = Identity()
```

ცვლილებების მიზანი იყო, რომ spatial information ძალიან სწრაფად არ დაკარგულიყო.

### Two-stage fine-tuning

საბოლოოდ გამოვიყენე two-stage fine-tuning:

#### Stage 1

პირველ ეტაპზე გავყინე ResNet feature extractor და დავატრეინინგე მხოლოდ classifier head:

```text
trainable layers: classifier only
learning rate = 0.001
```

ამ ეტაპზე classifier სწავლობდა FER-ის 7 emotion class-ს.

#### Stage 2

მეორე ეტაპზე გავხსენი ResNet-ის ბოლო layers:

```text
trainable layers: layer3 + layer4 + classifier
learning rate = 0.00005
```

ადრეული layers გაყინული დავტოვე, რადგან ისინი general visual features-ს ინახავენ. ბოლო layers კი სპეციფიურ features-ზე fine-tune-და.

### დამატებითი რეგულარიზაცია

ResNet ექსპერიმენტში გამოვიყენე:

```text
AdamW
weight decay = 5e-4
dropout = 0.4
label smoothing = 0.05
gradient clipping
learning rate scheduler
```

label smoothing დავამატე overconfidence-ის შესამცირებლად. gradient clipping დავამატე training-ის სტაბილურობისთვის.

საბოლოო test evaluation WandB-ზე დალოგილია ცალკე test run-ში.

### ანალიზი

ResNet18-მა აჩვენა, რომ transfer learning სასარგებლოა, მაგრამ პირდაპირი გამოყენება საკმარისი არ იყო. საჭირო გახდა ResNet-ის ადაპტაცია პატარა grayscale input-ისთვის.

Two-stage fine-tuning უკეთესი აღმოჩნდა, რადგან ჯერ classifier სწავლობდა ახალ task-ს, შემდეგ კი მხოლოდ ბოლო layers ერგებოდა FER dataset-ს. მიუხედავად ამისა, საუკეთესო შედეგი მაინც CNN-მა დააფიქსირა test data-ზე.

---

# Final Comparison

| Experiment    | Model                      |                                   Main Idea |            Test Accuracy | Main Observation                                   |
| ------------- | -------------------------- | ------------------------------------------: | -----------------------: | -------------------------------------------------- |
| Experiment 01 | Simple CNN                 |                              Baseline model |                  0.51312 | Pipeline worked, but model had limited capacity    |
| Experiment 02 | CNN + BatchNorm + Dropout  |                  Deeper and regularized CNN |                  0.58324 | Better feature extraction and more stable training |
| Experiment 03 | CNN + Augmentation         |                          Reduce overfitting |                  0.59601 | Train/validation gap became smaller                |
| Experiment 04 | VGG-style CNN              |       Stronger custom CNN feature extractor |                  0.64918 | Best custom CNN result                             |
| Experiment 05 | ResNet18 Transfer Learning | Pretrained model with two-stage fine-tuning |                  0.63524 | Best validation result overall                     |

---

# Overfitting და Underfitting ანალიზი

### Underfitting

პირველი მოდელი შედარებით underpowered იყო. SimpleCNN-ს ჰქონდა მცირე capacity და მხოლოდ ორი convolutional layer. ამიტომ validation accuracy დაახლოებით 50%-თან გაჩერდა.

ეს არ იყო ძლიერი underfitting იმ გაგებით, რომ მოდელი საერთოდ ვერ სწავლობდა, მაგრამ მისი capacity საკმარისი არ იყო რთული პატერნის დასაჭერად.

### Overfitting

Overfitting ყველაზე კარგად ჩანდა Experiment 04 და Experiment 05-ში.

Experiment 04-ში VGG-style CNN-ის train accuracy ბევრად მაღალი გახდა validation accuracy-ზე:

```text
train accuracy ≈ 72.59%
validation accuracy ≈ 62.98%
```

ეს აჩვენებდა overfitting-ს, მაგრამ validation accuracy მაინც მნიშვნელოვნად გაუმჯობესდა, ამიტომ მოდელის გაზრდა სასარგებლო იყო.

Experiment 05-ში ResNet კიდევ უფრო სწრაფად მიდიოდა overfitting-ისკენ, რადგან pretrained model-ს დიდი capacity აქვს.

### Generalization improvement

Experiment 03 იყო ყველაზე კარგი მაგალითი overfitting-ის შემცირების. data augmentation-ის შემდეგ training და validation accuracy ძალიან ახლოს იყო:

```text
train accuracy ≈ 58.60%
validation accuracy ≈ 56.87%
```

ეს აჩვენებდა, რომ augmentation-მა რეგულარიზატორის როლი შეასრულა და მოდელი დაასტაბილურა.

---

# Final Test Evaluation

თითოეული notebook-ის ბოლოს არის test section, რომელიც აკეთებს:

```text
1. checkpoint-ის ჩატვირთვა Google Drive-დან
2. forward sanity check
3. backward sanity check
4. internal test evaluation
5. confusion matrix logging
6. classification report logging
7. test loss და test accuracy logging WandB-ზე
```

ყველაზე მნიშვნელოვანი დასკვნები:

```text
პატარა CNN კარგი baseline იყო, მაგრამ შეზღუდული capacity ჰქონდა.
BatchNorm და Dropout training-ს ასტაბილურებს.
GAP ზედმეტად კუმშავდა spatial information-ს და ამ task-ზე ცუდად იმუშავა.
Data augmentation ამცირებს overfitting-ს.
VGG-style CNN უკეთ სწავლობს local facial features-ს.
ResNet18-ს სჭირდება ადაპტაცია 48×48 grayscale input-ზე.
Two-stage fine-tuning transfer learning-ს უფრო სტაბილურს ხდის.
Forward/backward sanity checks აუცილებელია pipeline-ის დასადასტურებლად.
```

---

# შეჯამება

პროექტში ეტაპობრივად გავტესტე რამდენიმე განსხვავებული neural network architecture facial expression recognition-ისთვის.

დავიწყე SimpleCNN baseline-ით, შემდეგ დავამატე BatchNorm, Dropout და დამატებითი convolutional block. ამის შემდეგ გამოვცადე data augmentation და class imbalance handling. შემდეგ ავაგე VGG-style CNN, რომელმაც custom CNN-ებში საუკეთესო შედეგი აჩვენა. ბოლოს გამოვიყენე ResNet18 transfer learning და two-stage fine-tuning, რომელმაც საუკეთესო validation accuracy მიიღო.

ყველა ექსპერიმენტი დალოგილია WandB-ზე და შედეგები შეჯამებულია WandB report-ში.
