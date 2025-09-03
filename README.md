# Sentiment Analysis Platform

A modular, full-stack sentiment analysis application that demonstrates foundational and advanced NLP techniques. Built with a scalable backend, interactive React frontend, and integrated analytics logging, this project is designed for real-world deployment and resume impact.

## 🚀 Project Overview

This platform allows users to input text and receive sentiment predictions using multiple NLP models. It supports emoji-based feedback, tracks user interactions, and logs analytics for model performance and usage trends.

### 🔍 Features

- **Multi-model sentiment analysis**:
  - TF-IDF + SVM
  - LSTM (Keras)
  - Transformer (DistilBERT via HuggingFace)
- **Modular backend architecture** (NestJS + Prisma)
- **Interactive frontend** (React + TypeScript)
- **Emoji-based feedback loop**
- **Analytics logging** for model usage and sentiment trends
- **Model selection toggle** for comparative evaluation

## 🧱 Tech Stack

| Layer       | Technologies Used                                  |
|------------|-----------------------------------------------------|
| Frontend   | React, TypeScript, Axios                            |
| Backend    | NestJS, Prisma ORM, PostgreSQL                      |
| ML Models  | scikit-learn, Keras, PyTorch, HuggingFace Transformers |
| DevOps     | Docker, GitHub Actions (optional)                   |

## 📦 Installation

### Prerequisites

- Node.js (v18+)
- Python (v3.8+)
- PostgreSQL
- Docker (optional for containerization)
- **Must Download dataset `sentiment140` from [Kaggle dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and place the csv file in `backend\dataset\sentiment140.csv`**

### Backend Setup

```bash
cd backend
npm install
npx prisma generate
npx prisma migrate dev
npm run start:dev
```

### Backend Setup

```bash
cd frontend
npm install
npm start
```
### ML Model Server

```bash
cd ml-models
# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```


## 📊 Analytics & Logging

- Logs model usage frequency
- Tracks sentiment distribution over time
- Captures emoji feedback for UI/UX insights


## 🧠 Model Details

| Model        | Description                                                 |
| ------------ | ----------------------------------------------------------- | 
| TF-IDF + SVM | Lightweight baseline using scikit-learn                     | 
| LSTM         | Sequential model trained on IMDB dataset                    | 
| Transformer  | Fine-tuned DistilBERT for robust sentiment detection        |



## 📁 Folder Structure

```
sentimentAnalysis/
├── backend/         # NestJS API with Prisma
├── frontend/        # React UI
├── ml-models/       # Python ML services
├── prisma/          # DB schema and migrations
└── README.md
```

## Future Enhancements
- Add user authentication
- Deploy via Docker Compose
- Integrate real-time feedback loop for model retraining
- Expand to multilingual sentiment analysis

