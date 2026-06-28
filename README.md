# Student Marks Prediction

## Overview

Student Marks Prediction is a Flask-based web application designed for a school-themed environment, ideal for parent-teacher meeting use cases and student performance insights. The app lets users enter student details, predicts Maths marks using a trained machine learning model, and displays results clearly on the same page.

## Screenshots

![Homepage](ProjectScreenshots/Screenshot%202026-06-28%20223051.png)

![Prediction Form](ProjectScreenshots/Screenshot%202026-06-28%20223317.png)

![Results](ProjectScreenshots/Screenshot%202026-06-28%20223330.png)

## Features

- User-friendly web interface for easy navigation
- Input form for student details and academic variables
- Prediction of Maths marks using a regression model
- Results displayed on the same page for quick review

## Model Details

The project uses a regression-based machine learning model to predict student Maths marks. The model pipeline includes:

- Handling categorical variables through encoding
- Applying `StandardScaler` for normalization
- Splitting data into train and test sets
- Training a regression model to estimate marks

Model accuracy is approximately **79%** on the test data.

## Setup & Commands

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If Flask or any required library is missing, add it with:

```bash
pip install Flask
```

Or install a specific package manually:

```bash
pip install scikit-learn pandas numpy
```

3. Run the Flask app:

```bash
python app.py
```

4. Git workflow commands:

```bash
git clone <repository-url>
cd <repository-folder>
git add .
git commit -m "Add Student Marks Prediction project"
git push origin main
```

## Usage

1. Activate the virtual environment.
2. Install dependencies.
3. Start the app with `python app.py`.
4. Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

5. Fill in student details and submit the form to view the predicted Maths mark.

## Future Improvements

- Add charts and visual performance reports for each student
- Provide summary dashboards for class-level insights
- Improve prediction accuracy with advanced models such as ensemble regressors or gradient boosting
- Expand input options to include attendance, study habits, and parent engagement data

## Notes

This project is built to support a school-style performance review workflow, making it useful for teachers, parents, and administrators looking to understand student Maths performance quickly.
