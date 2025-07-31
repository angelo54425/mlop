from locust import HttpUser, task, between
import random
import json

class LoadTestUser(HttpUser):
    wait_time = between(1, 5) 

    @task(5)  # Higher weight = more frequent
    def predict(self):
        payload = {
            "Age": random.uniform(20, 70),
            "Weight": random.uniform(50, 100),
            "BP_History": random.randint(0, 1),
            "Medication": random.randint(0, 1),
            "Family_History": random.randint(0, 1),
            "Exercise_Level": random.randint(0, 2),
            "Smoking_Status": random.randint(0, 1),
            "Cholesterol_Level": random.randint(0, 2),
            "Salt_Intake_Level": random.randint(0, 2),
            "Alcohol_Consumption": random.randint(0, 2)
        }
        self.client.post("/predict/", json=payload)

    @task(2)
    def retrain(self):
        with open("sample_data.csv", "rb") as file:
            self.client.post(
                "/retrain/",
                files={"file": ("sample_data.csv", file, "text/csv")}
            )

    @task(1)
    def fine_tune(self):
        with open("sample_data.csv", "rb") as file:
            self.client.post(
                "/fine_tune/",
                files={"file": ("sample_data.csv", file, "text/csv")},
                data={"epochs": random.randint(5, 10)}
            )
