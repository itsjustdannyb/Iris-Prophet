from pydantic import BaseModel
from fastapi import Form

# iris data specs
class Iris_data_specs(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

    # input validation with pydantic
    @classmethod
    def as_form(
    cls, 
    sepal_length_cm: float = Form(...),
    sepal_width_cm: float = Form(...),
    petal_length_cm: float = Form(...),
    petal_width_cm: float = Form(...),
    ):
        return cls(
            sepal_length_cm=sepal_length_cm,
            sepal_width_cm=sepal_width_cm,
            petal_length_cm=petal_length_cm,
            petal_width_cm=petal_width_cm
        )

