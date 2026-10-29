
# Solar Cell Surface Defect Detection Using Optimized YOLOv5

## 📌 Project Overview

This project focuses on detecting surface defects in solar cells using deep learning and computer vision techniques. It provides a web-based application for loading datasets, processing images, training models, and predicting defects in solar-cell images.

The objective is to explore automated solar-cell quality inspection using deep-learning algorithms.

## 🎯 Objectives

- Detect surface defects in solar-cell images.
- Apply deep learning and computer vision techniques.
- Process and prepare solar-cell image datasets.
- Compare model performance using evaluation metrics.
- Build a web-based interface for solar-cell defect prediction.

## ✨ Features

- User registration and login.
- Load and process the ELPV dataset.
- Split images into training and testing datasets.
- Train and evaluate defect-detection models.
- Upload solar-cell images for prediction.
- Display predicted defect classes and probabilities, where supported by the model.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Web Framework:** Django
- **Database:** MySQL
- **Development Environment:** Visual Studio Code

## 📂 Project Structure

```text
SolarCell/
├── Solar/                  # Django project configuration
├── SolarApp/               # Application modules
├── model/                  # Model files, if available
├── testImages/             # Test images, if included
├── Attention.py            # Custom attention layer
├── manage.py               # Django management script
├── requirements.txt        # Python dependencies
├── database.txt            # Database setup instructions
├── .gitignore
└── README.md
```

The actual files and folders may vary depending on the repository version.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Praveen-1131/Solar-Cell-Surface-Defect-Detection-YOLOv5.git
cd Solar-Cell-Surface-Defect-Detection-YOLOv5
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If your terminal already displays `(.venv)`, the environment is active.

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

**Note:** The project uses legacy dependency versions. A compatible Python environment may be required to install and run all packages successfully.

### 5. Configure MySQL

The `database.txt` file contains SQL commands for creating the `solar` database and the `register` table.

Execute the required SQL commands in MySQL and ensure that the Django database configuration matches your local database settings.

Do not upload passwords or other sensitive credentials to GitHub.

### 6. Run Database Migrations

```powershell
python manage.py migrate
```

### 7. Run the Application

```powershell
python manage.py runserver
```

After the server starts, open the following URL in your browser:

http://127.0.0.1:8000/index.html

Keep the terminal open while using the application.

## 🧪 Testing

1. Start the Django development server.
2. Open the application in your browser.
3. Register or log in, if configured.
4. Load and process the dataset, if available.
5. Upload a suitable solar-cell test image.
6. Inspect the predicted defect class and probability.

Actual model performance should be verified through repeatable experiments.
```

## 🔮 Future Improvements

- Improve defect detection accuracy and robustness.
- Evaluate precision, recall, F1-score, and other performance metrics.
- Improve prediction visualization.
- Test the model on additional solar-cell images.
- Enhance the application's usability and reporting features.

## 👨‍💻 Author

**Praveen Venkata Bhargav**

GitHub: https://github.com/Praveen-1131

## 📄 Disclaimer

This project was developed for academic purposes to explore deep learning and computer vision for solar-cell surface-defect detection. Its predictions should not be treated as validated industrial quality-control results without appropriate testing and validation.
