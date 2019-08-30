# ESP-Training

1. ESP_Predictor.ipynb, ESP_Prescriptor.ipynb, these files will train predictor and presciptor model with 5 context variables , 2 action variables , 1 outcome variable.
From the training data, the predcitor model is trained and saved. Further the predcitor model is used to calculate fitness value of each candidate while training the Prescriptor models.


2. In ESP_Custom.ipynb instead of using a predictor model, predefined criterion or weights are used to calculate the Fitness value using the script getFitness.py.


