# SageMaker Model Deployment with Hugging Face – Error Investigation

This document details the process of deploying a Hugging Face pre-trained model on AWS SageMaker Studio using JupyterLab. It includes key setup steps, configuration requirements, and an analysis of an error encountered during deployment.

## Table of Contents
- [Key Components](#key-components)
- [Requirements](#requirements)
- [Steps to Reproduce](#steps-to-reproduce)
  - [1. Select the Pre-trained Model](#1-select-the-pre-trained-model)
  - [2. Download Model Files](#2-download-model-files)
  - [3. Upload to S3](#3-upload-to-s3)
  - [4. SageMaker Studio Setup](#4-sagemaker-studio-setup)
  - [5. Model Deployment Code](#5-model-deployment-code)
- [Encountered Error](#encountered-error)
- [Analysis of Error](#analysis-of-error)
- [Next Steps](#next-steps)

## Key Components
- **Transformers Library**: Hugging Face's library containing pre-trained models for a variety of machine learning tasks.
- **HuggingFaceModel Class (SageMaker SDK)**: A utility class in the SageMaker SDK that simplifies the process of loading and deploying Hugging Face models on SageMaker.

## Requirements
- **AWS IAM Role**: Requires permissions for SageMaker, S3, and Hugging Face model access.
- **S3 Bucket**: Storage for model files.
- **AWS SageMaker**: Configured environment for deploying the model.

## Steps to Reproduce

### 1. Select the Pre-trained Model
The selected model and task for this deployment are:
- **Model ID**: `llava-hf/llama3-llava-next-8b-hf`
- **Task**: Image-to-Text Translation

### 2. Download Model Files
Manually downloaded the following model and configuration files:
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`
- `chat_template.json`
- `generation_config.json`
- `preprocessor_config.json`
- `tokenizer.json`
- `config.json`
- `model.safetensors.index.json`
- `special_tokens_map.json`
- `tokenizer_config.json`

### 3. Upload to S3
The model files were compressed and uploaded manually to the following S3 bucket:
- **S3 Bucket**: `sagemaker-us-east-2-476671003699`
- **Folder**: `llama3/`

### 4. SageMaker Studio Setup
SageMaker Studio was set up with the following instance type:
- **Instance Type**: `ml.m5.2xlarge`

#### Environment Setup
1. **Global Environment**: Encountered an error while installing the `sagemaker` package using `!pip install sagemaker -U`. Refer to the image `pip_install_sgmkr` for details.
2. **Virtual Environment**: 
   - Created a virtual environment using `!python3 -m venv`.
   - Installed required packages: `!pip install sagemaker -U`, `!pip install transformers`.

### 5. Model Deployment Code

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface.model import HuggingFaceModel
import boto3

try:
    role = sagemaker.get_execution_role()
except:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName="sagemaker_execution_role")['Role']['Arn']

# Define S3 path for the model
model_s3_path = "s3://sagemaker-us-east-2-476671003699/llama3/model.tar.gz"

huggingface_model = HuggingFaceModel(
    model_data=model_s3_path,
    role=role,
    transformers_version="4.12",
    pytorch_version="1.9",
    py_version="py38",
)

# Deploy the model
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.2xlarge"
)

# Sample data for inference
data = {
   "inputs": "who are you"
}

# Invoke the endpoint for prediction
predictor.predict(data)
