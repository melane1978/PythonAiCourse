# How to Use CrimeML AI


This AI is programmed in python and using Pytorch Neural Network model.
It's job is to predict how much a Crime against a Person, will increase or decrease in 2024 for each region and month in Stockholm, Sweden.

Here is how you use it:

1. Create a python .venv with:  python -m venv .venv
2. Source into the virtual Env: source .venv/bin/activate
3. Install requirements: pip3 install -r requirements.txt


# Testing CrimeML AI.

1. Run the CrimeML1.py script, you can change epocs to what you want in the script, its on 10000 now.

2. When its done, if you want to see the result, run the pythonscript
predict2024v1.py. That script fix the data to be readable and open a menu in your terminal. There you can choose which Region and crime you want to see, the predicted values against 2023 data.
You end the scrip by closing plot and press 0

I have saved all models for each region under saved_models.

Have fun and enjoy!

melane@gmail.com
