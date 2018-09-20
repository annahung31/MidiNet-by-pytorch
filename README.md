
This is an Implementation of MidiNet by pytorch.

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

dataset is from theorytab : https://www.hooktheory.com/theorytab 

You can find crawler here : https://github.com/wayne391/Symbolic-Musical-Datasets 


requirement.py                  |  toolkits used in the whole work


get_data.py                     |  get melody and chord matrix from xml


get_train_and_test_data.py      |  seperate the data into training set and testing set


ops.py                          |  some functions used in model


model.py                        |  Generator and Discriminator.   (Based on model 3 in the MidiNet paper)


main.py                         |  training setting, drawing setting, generation setting.


demo.py                         |  transform matrix into midi. (input : melody and chord matrix, output : midi)




