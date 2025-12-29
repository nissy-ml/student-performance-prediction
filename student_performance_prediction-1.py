# student_performance_prediction.py

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# CSV data as a string
csv_data = """
StudentID,StudyHours,Attendance,PreviousScore,FinalScore
1,5,80,70,75
2,3,60,65,60
3,8,90,85,90
4,2,50,55,50
5,7,85,80,85
6,4,70,60,65
7,6,75,78,80
8,9,95,90,95
9,1,40,50,45
10,5,80,75,78
"""

# Read the CSV data from the string
data = pd.read_csv(StringIO(csv_data))

# Display first few rows
print("Dataset:")
print(data.head())

# Features and target
X = data[['StudyHours', 'Attendance', 'PreviousScore']]
y = data['FinalScore']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Predicting a new student
new_student = pd.DataFrame({
    'StudyHours': [6],
    'Attendance': [85],
    'PreviousScore': [78]
})

predicted_score = model.predict(new_student)
print(f"\nPredicted Final Score for the new student: {predicted_score[0]:.2f}")