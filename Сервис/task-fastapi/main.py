from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

#pip install uvicorn[standard]
#uvicorn main:app --reload

app = FastAPI()

model, MSE, r2_score, coef, intercept, enc, scaler = pickle.load(open("tuple_model_best.pkl", 'rb'))

class Item(BaseModel):

    year: int
    max_torque_rpm: int
    km_driven: int
    mileage: int
    engine: int
    max_power: int
    torque: int
    max_power_engine: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    seats: str

@app.post('/predict_item')
async def input_car(new_car: Item):
    X_cat = pd.DataFrame({'fuel': [new_car.fuel], \
                            'seller_type': [new_car.seller_type], \
                            'transmission': [new_car.transmission], \
                            'owner': [new_car.owner], \
                            'seats': [new_car.seats] \
                             })

    X_num = pd.DataFrame({'year': [new_car.year], \
                            'yearsq': [new_car.year**2], \
                            'max_torque_rpm': [new_car.max_torque_rpm], \
                            'km_driven': [new_car.km_driven], \
                            'mileage': [new_car.mileage], \
                            'engine': [new_car.engine], \
                            'max_power': [new_car.max_power], \
                            'torque': [new_car.torque], \
                            'max_power_engine': [new_car.max_power_engine] \
                             })

    X_cat = pd.DataFrame(enc.transform(X_cat).toarray())
    X_dum = pd.concat([X_num, X_cat], axis=1, join='inner')
    X_dum = scaler.transform(X_dum)
    pred = model.predict(X_dum)

    return pred[0][0]