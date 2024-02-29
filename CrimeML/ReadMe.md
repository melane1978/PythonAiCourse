# How to Use CrimeML AI


This AI is programmed in python and using Pytorch Neural Network model.
It's job is to predict how much crime against a citizen will increase or decrease in 2024 for each month in the counties of  Stockholm, Sweden.

Here is how you use it:

1. Create a python .venv with:  python -m venv .venv
2. Source into the virtual Env: source .venv/bin/activate
3. Install requirements: pip3 install -r requirements.txt


# Testing CrimeML AI.

1. Run the CrimeML1.py script under TestingML, you can change epocs to what you want in the script, its on 10000 now.

2. When its done, if you want to see the result, run the pythonscript
predict2024v1.py. That script fixes the data to be readable and open a GUI menu.
There you can choose which county and which crime you want to view which includes the predicted values against 2023 data.     

3. You can also run a menu from the Terminal if the GUI does not work.
Just run the plot.py, choose county, then the crime and a plot will show up.     
To close the script, close the Plot and press 0

I have saved all models for each region under saved_models.

Have fun and enjoy!

# Reflections

This was a fun but difficult project, it took more time to get the data in the right format than building the AI in pytorch.
Thanks to Jupyter it's easier to redue and fix the data, after 100 times of trying to get it right.
It was also difficult to know which values I should focus on: month data, year data, or 100000 persons per capita.
I also needed to prepare the data for 2023 to be in the same format as the predicted value for 2024, to have something 
to compare against in a PLOT chart.

My Pytorch NN module is working great I think when running like 10000 Epochs I have a loss around 10-20%, I tried to run it 
with 1M Epochs, but that made it overthink and the values result was all over the place.
So the best epoch for this AI is around 10.000 because then I think it is without errors.           
We are unable to predict the data for crime in 2024, therefore a bit difficult to know what is accurate,
and what is wrong.

I have also not attempted to configure any advanced loss function: I just used the default setup (nn.MSELoss()), this approach seems to work ok.

I tried to change the learning rate for the optimizer, but could not see very much diffrence in the predicted data.

So how the NM module is set up now appears to be the best practies for this AI.

I was intrested to use Pytorch CUDA memory, to see if my AI will perfom better, but do to my computer being a Mac it does not support CUDA. Therefore I needed to use the CPU instead, but for a much smaller AI I concluded that this was not a problem.

My final thoughts; can this predicted data be correct? No, it can not, but it seems to follow a good pathern and 
the loss is not very high.

Version 1.0 CrimeML AI, 2024-02-29
melane1978@gmail.com
