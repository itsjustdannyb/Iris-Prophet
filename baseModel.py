from pydantic import BaseModel

# iris data specs
class Iris_data_specs(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float
