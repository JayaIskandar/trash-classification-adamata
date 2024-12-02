# 🗑️ Trash Classification Models: CNN from Scratch and ResNet50
# ✨ Introduction
This repository is part of my internship test at Adamata for the AI Engineer role. The task was to classify trash images into six categories: paper, glass, plastic, metal, cardboard, and trash.

**Dataset: 
5.05K+ images <br>
paper 23.5% <br>
glass 19.8% <br>
plastic 19.1% <br>
metal 16.2% <br>
cardboard 15.9% <br>
trash 5.4%**

However, I decided to implement **two different models** to compare their performance:

**1.  A pure Convolutional Neural Network (CNN) built from scratch - using PyTorch. <br>
2. A ResNet50 model fine-tuned using transfer learning - using TensorFlow.**
   
From these 2 models, I have explored the tradeoffs as well. The dataset used for this task is the TrashNet dataset (https://huggingface.co/datasets/garythung/trashnet).

<br> 

# 📊 W&B and Hugging Face Results
## Access the W&B Results:
Pure CNN:  https://wandb.ai/jayaiskandar17-individual/trashnet-classification **(choose the young-lake-6)** <br>
ResNet: https://wandb.ai/jayaiskandar17-individual/trash-classification-o **(choose the legendary-dawn-3)**

## Access the Hugging Face Models:
Pure CNN: https://huggingface.co/jayaiskandar17/trash-classification-pure-cnn/tree/main <br>
ResNet: https://huggingface.co/jayaiskandar17/trash-classification-resnet/tree/main

Model files can be downloaded from the site above^

<br> 

# 🧠 **Justifications for Model Selection**
## **1. CNN from Scratch** <br>
**Pros:** <br>
- Provides full control over the architecture and design.  <br>
- Performed better in real-world predictions despite lower training accuracy. <br>

**Cons:** <br>
- Requires significant computational resources and time for training **(7 hours on Google Colab Pro L4 GPU for 20 epochs)**.  <br>
- Achieved lower accuracy (57.66%) during training/validation compared to ResNet.  <br>

## **2. ResNet50** <br>
**Pros:** <br>
- Using pre-trained weights, significantly reducing training time **(100 minutes for 10 epochs)**.  <br>
- Achieved 98% accuracy within 10 epochs on the training dataset. <br>

**Cons:** <br>
- Can struggle with domain-specific generalization if the pre-trained weights are not well-suited for the target dataset.

<br> 

# 📝 **SUMMARY NOTES**
#### **When comparing the performance of these two models, the ResNet model clearly performed better at accurately classifying items, such as identifying Trash as Trash. While the custom CNN often showed higher confidence in its predictions, this confidence was misleading and highlighted its tendency to overfit to specific patterns in the training data. For example, the CNN mistakenly labeled Cardboard as Glass with 100% confidence—a clear sign of overfitting.**

#### **This issue becomes even more apparent when looking at the evaluation graphs below. The ResNet model shows a more stable and balanced learning curve, which indicates it generalizes better to new data. On the other hand, the CNN's graph was bit rough.**

## **PURE CNN**
![image](https://github.com/user-attachments/assets/3b50401a-0559-4043-adcf-00a034e32acf)
![image](https://github.com/user-attachments/assets/be4abe78-bb45-4044-8271-0b844422093d)


## **RESNET**
![image](https://github.com/user-attachments/assets/6222dca1-1b18-4e66-99cd-9a44b78f961a)
![image](https://github.com/user-attachments/assets/b5d7b36e-d061-4e67-95dc-1516a107364d)

<br> 

# 🛠️ **How to Run**
## **1. Clone the Repository**
```git clone https://github.com/JayaIskandar/trash-classification-adamata.git``` <br>
```cd trash-classification-adamata```

## **2.A Run the pure CNN model**
```cd trash-classification-pure-cnn``` <br>
```pip install -r requirements.txt``` <br>
```jupyter notebook trash_classification_pure_cnn.ipynb```

## **2.B Run the ResNet50 model**
```cd trash-classification-resnet``` <br>
```pip install -r requirements.txt``` <br>
```jupyter notebook trash_classification_resnet.ipynb```

<br>

# Github Actions
## Setting Up API Keys
if cloned:
1. **Weights & Biases (W&B)**:
   - Obtain your API key from [wandb.ai](https://wandb.ai/settings).
   - Set it as an environment variable:
     ```bash
     export WANDB_API_KEY=your_api_key
     ```

2. **Hugging Face**:
   - Obtain your token from [Hugging Face Hub](https://huggingface.co/settings/tokens).
   - Set it as an environment variable:
     ```bash
     export HF_TOKEN=your_huggingface_token
     ```
     
Or if forked:
1. Go to your forked repository on GitHub.
2. Navigate to **Settings > Security > Secrets and variables > Actions**.
3. Add the following secrets:
   - `WANDB_API_KEY`: Your W&B API key.
   - `HF_TOKEN`: Your Hugging Face token.
4. Push or trigger the workflow.
