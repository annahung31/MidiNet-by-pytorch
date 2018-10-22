
This is an Implementation of MidiNet by pytorch.

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

dataset is from theorytab : https://www.hooktheory.com/theorytab 

You can find crawler here : https://github.com/wayne391/Symbolic-Musical-Datasets 




--------------------------------------------------------------------------------------------------
Prepare the data

get_data.py                     |  get melody and chord matrix from xml


get_train_and_test_data.py      |  seperate the melody data into training set and testing set (chord preparation not included)

--------------------------------------------------------------------------------------------------
After you have the data, 
1. Make sure you have toolkits in the requirement.py
2. Run main.py ,  
  is_train = 1 for training, 
  is_draw = 1 for drawing loss, 
  is_sample = 1 for generating music after finishing training.
  
3. If you would like to turn the output into real midi for listening
  Run demo.py

--------------------------------------------------------------------------------------------------
requirement.py                  |  toolkits used in the whole work

main.py                         |  training setting, drawing setting, generation setting.

ops.py                          |  some functions used in model

model.py                        |  Generator and Discriminator.   (Based on model 3 in the MidiNet paper)

demo.py                         |  transform matrix into midi. (input : melody and chord matrix, output : midi)




