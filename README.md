# Solar Cell Surface Defect Detection Using Optimized YOLOv5

## 📌 Project Overview

This project focuses on detecting surface defects in solar cells using deep learning and computer vision techniques. It provides a web-based interface for interacting with the system, loading the dataset, training the model, and testing defect predictions on solar cell images.

The project aims to support solar cell quality inspection by applying automated image analysis.

## 🎯 Objectives

- Detect defects in solar cell images using deep learning.
- Explore optimized YOLO-based approaches for defect detection.
- Provide a web interface for dataset loading, model training, and image prediction.
- Demonstrate the application of computer vision in solar panel quality inspection.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Data Analysis:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Web Framework:** Django
- **Database:** MySQL
- **Development Environment:** Visual Studio Code

## 📂 Project Structure

```text
SolarCell/
├── model/              # Model files (large trained models excluded from GitHub)
├── Solar/              # Django project settings and URL configuration
├── SolarApp/           # Application views, models and URLs
├── testImages/         # Images used for testing
├── Attention.py        # Custom attention layer
├── manage.py           # Django management script
├── requirements.txt    # Python dependencies
├── database.txt        # Database setup reference
├── .gitignore          # Files excluded from version control
└── README.md           # Project documentation
```

## ⚙️ Features

- User registration and login.
- Dataset loading and processing.
- Model training interface.
- Solar cell image upload and prediction.
- Web-based interaction using Django.

## 🚀 Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Praveen-1131/Solar-Cell-Surface-Defect-Detection-YOLOv5.git
cd Solar-Cell-Surface-Defect-Detection-YOLOv5
```

### 2. Create a virtual environment

Use a compatible Python version for this legacy project.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

**Note:** The project uses older dependency versions. Installation may require a compatible Python environment and adjustments to package versions.

### 4. Configure the database

Create a MySQL database named `solar` and configure the required `register` table using the SQL reference in `database.txt`.

Configure database credentials securely. Do not commit passwords or other secrets to GitHub.

### 5. Run database migrations

```bash
python manage.py migrate
```

### 6. Start the Django development server

```bash
python manage.py runserver
```

Open the application in your browser:

http://127.0.0.1:8000/index.html

## 🧪 Testing

1. Start the Django server.
2. Open the web application.
3. Register an account and log in.
4. Load the dataset and access the model training page.
5. Open the prediction page, upload a suitable solar cell image, and inspect the displayed result.

Actual prediction accuracy and model performance should be evaluated using experimental results.

## 🔮 Future Improvements

- Improve defect detection accuracy and robustness.
- Add evaluation metrics such as precision, recall, and F1-score.
- Provide visual explanations of detected defects.
- Improve the user interface and prediction reporting.
- Deploy the application for wider accessibility.

## 👨‍💻 Author

**Praveen Venkata Bhargav**

GitHub: [Praveen-1131](https://github.com/Praveen-1131)

## 📄 Disclaimer

This project was developed for academic purposes to explore deep learning and computer vision for solar cell defect detection.