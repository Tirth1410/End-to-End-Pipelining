
# 🚀 **Azure Machine Learning: Step-by-Step Pipelining**  

Welcome to the **Azure Machine Learning: Step-by-Step Pipelining** repository! 🎉 This repository is your one-stop guide to learning how to build and deploy machine learning pipelines using Azure ML. The codes are 100% reproducible and it is beginner-friendly.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/f277cfa9-c2d9-4a6d-9766-e60e42b4a102" alt="Azure ML Pipeline Illustration" width="700" />
</p>


---

## 📂 **Repository Structure**  

Here's a quick look at the repository contents:  

```plaintext
📦 Repository
├── 📂 model
│   ├── 📄 new_model_state_dict.pth
│   ├── 📄 vocab.txt
├── 📂 scripts
│   ├── 📄 score.py
│   ├── 📄 script1.py
│   ├── 📄 script2.py
│   ├── 📄 script3.py
│   ├── 📄 script4.py
├── 📂 workspace
│   ├── 📄 dataset.csv
│   ├── 📄 env.yml
├── 🖼️ Pipeline_Structure.png
├── 📄 README.md
└── 📄 readme_MLOps.ipynb

```

## 📂 Folder Overview

### 📂 `model`
- Contains pre-trained models and vocabulary files.
  - 📄 `new_model_state_dict.pth`: The saved state dictionary of the model.
  - 📄 `vocab.txt`: Vocabulary used by the model.

### 📂 `scripts`
- Includes Python scripts for various stages of the pipeline.
  - 📄 `score.py`: Script to evaluate the model's performance.
  - 📄 `script1.py`: First stage of the CI pipeline (e.g., data preprocessing).
  - 📄 `script2.py`: Second stage of the CI pipeline (e.g., model training).
  - 📄 `script3.py`: Third stage of CI pipeline (e.g., old and new model (the one after training) comparison).
  - 📄 `script4.py`: The first and only stage of the CD Pipeline that is triggered only if new model is better than the old one.

### 📂 `workspace`
- Workspace-related files like datasets and environment configurations.
  - 📄 `dataset.csv`: Dataset used for training and evaluation.
  - 📄 `env.yml`: Conda environment file to set up dependencies.

### 🖼️ `Pipeline_Structure.png`
- Diagram illustrating the pipeline's structure or workflow.

### 📄 `README.md`
- Main documentation file providing an overview of the repository.

### 📄 `readme_MLOps.ipynb`
- A Jupyter Notebook used to create this project and 100% reproducible.

---

## 🌟 **Prerequisites**  

Before you start, make sure you have the following installed and understand the basics:  

### 📦 Required Libraries:  
- **PyTorch**: For building and training neural networks.  
- **NumPy**: For numerical operations and data manipulation.  

### 📚 Concepts to Know:  
- **Training and Testing Models**: Fit models to training data and evaluate them on test data.  
- **API Endpoints**: Deploy models as RESTful APIs for real-time inference.  

---

## 🤔 **Why Use Pipelines in AI/ML?**  

Pipelines simplify and automate the machine learning lifecycle, addressing challenges like:  
- **Seamless Model Updates**: Automate retraining and deployment without manual intervention.  
- **Access Control**: Restrict changes to specific workflow stages.  
- **Scalability**: Handle growing datasets and complex models.  

> **Example**: Need to deploy an updated model? Pipelines ensure the transition is smooth, avoiding downtime or conflicts.

---

## 🖼️ **Visual Workflow**  

<p align="center">
  <img src="https://github.com/Hyperspectral01/AzureML_Step-by-Step_Pipelining/blob/main/Pipeline_Structure.png" alt="Pipeline Steps" width="700" />
</p>  

---

## 🛠️ **Steps Involved in the Project**  

### 1️⃣ **Setup Workspace and Compute Resources**  
- Set up an **Azure workspace** and provision compute resources (e.g., Azure Virtual Machines) for data processing and model training.  
- These resources enable efficient handling of large-scale operations.  

### 2️⃣ **Upload Files and Resources**  
- Upload the pretrained model and `vocab.txt` to the Azure workspace.  
- Ensure these resources are available for future training and tokenization tasks.  

### 3️⃣ **Connect to Workspace**  
- Establish a secure connection to the Azure workspace using the **Azure ML SDK**.  
- This allows seamless access to Azure resources for model management and machine learning tasks.  

### 4️⃣ **Model Registration**  
- Register the pretrained model within Azure ML to facilitate:  
  - Versioning  
  - Easy access  
  - Future training or deployment steps  

### 5️⃣ **Upload Vocabulary File**  
- Upload the `vocab.txt` file to the workspace.  
- This file contains the vocabulary needed for tokenizing input data and training the model.  

### 6️⃣ **Dataset Preparation**  
- Store `dataset.csv` in **Azure Blob Storage** for easy access during preprocessing and model training.  
- Blob Storage ensures efficient handling of large-scale data.  

### 7️⃣ **script1.py: Data Preparation**  
- Develop `script1.py` to preprocess the dataset:  
  - Tokenize input data  
  - Extract features  
  - Split into training (`train_x`, `train_y`) and testing (`test_x`, `test_y`) sets  
- Save the preprocessed data for use in subsequent pipeline steps.  

### 8️⃣ **script2.py: Model Training**  
- Create `script2.py` to fine-tune the pretrained model using the training data (`train_x`, `train_y`).  
- Prepare the trained model for evaluation with test data (`test_x`, `test_y`).  

### 9️⃣ **script3.py: Model Comparison and Evaluation**  
- Implement `script3.py` to evaluate both old and new models using the test data (`test_x`, `test_y`).  
- Record performance in an output file:  
  - `True`: New model performs better  
  - Otherwise, `False`  

### 🔟 **Pipeline for Model Training and Evaluation**  
- Design a pipeline to automate:  
  - Model training  
  - Evaluation  
  - Comparison  
- Ensure reproducibility and integrate it into a **CI/CD workflow** for efficient updates.  

### 1️⃣1️⃣ **score.py: Web Input Processing and Prediction**  
- Develop `score.py` to handle incoming data from web applications:  
  - Tokenize text using `vocab.txt`  
  - Feed it to the trained model  
  - Generate predictions  
- Designed for real-time inference in production environments.  

### 1️⃣2️⃣ **script4.py: Deployment Script**  
- Write `script4.py` to automate the deployment of the latest model to an **Azure endpoint**.  
- Ensure the model is ready to serve predictions via web requests.  

### 1️⃣3️⃣ **Pipeline for Deployment**  
- Build a deployment pipeline that automates:  
  - Model registration  
  - Deployment scripts  
  - Scaling capabilities  
- Ensure efficient and seamless deployment.  

### 1️⃣4️⃣ **Function: MLOps**  
- Develop the `MLOps` function to orchestrate:  
  - Training pipeline  
  - Deployment pipeline  
- By passing the dataset name as an argument, this function ensures smooth execution of the entire ML lifecycle.  

---

## 📊 **Results : Using the Endpoint** 

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e67f6a7-8aea-42a1-88a4-103f38932578" alt="Results" width="700" />
</p>

## 🎯 **Key Benefits**  

✅ **Automation**: Reduces errors and ensures reproducibility.  
✅ **Versioning**: Track datasets, models, and workflows.  
✅ **Scalability**: Handles large-scale operations efficiently.  
✅ **Collaboration**: Supports team-based workflows with access control.  

---

## 🔗 **Extra Links**  

- [What is MLOPS? (lang:Hindi)](https://www.youtube.com/watch?v=6SRifO6dmuE&t=664s&pp=ygUNd2hhdCBpcyBtbG9wcw%3D%3D)
- [Brief Idea about Devops in Azure (lang:en)](https://www.youtube.com/watch?v=4BibQ69MD8c&t=71s)  
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)  
- [PyTorch Official Website](https://pytorch.org/)  
- [NumPy Official Documentation](https://numpy.org/doc/)  

---

🎉 Thank you for exploring this project! Let's make ML workflows efficient and scalable with Azure Pipelines. 🚀  

