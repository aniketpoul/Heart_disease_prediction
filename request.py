import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'BMI':1, 'Smoking':1, 'AlcoholDrinking':1,'Stroke':1,'PhysicalHealth':1,'MentalHealth':1,'DiffWalking':1,
                            'Sex':1,'AgeCategory':1,'PhysicalActivity':1,'GenHealth':1,'SleepTime':1,'Asthma':1,'KidneyDisease':1,'SkinCancer':1})

print(r.json())