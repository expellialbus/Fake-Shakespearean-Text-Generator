# Fake Shakespearean Text Generator

This project contains an impelementation of stateful Char-RNN model to generate fake shakespearean texts.

Files and folders of the project.
> [models](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/tree/master/models) folder

This folder contains to zip file, one for stateful model and the other for stateless model (this model files are fully saved model architectures,not just weights).

> [weights.zip](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/weights.zip)

As you its name implies, this zip file contains the model's weights as checkpoint format (see [tensorflow model save formats](https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights)).

> [tokenizer.save](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/tokenizer.save)

This file is an saved and trained (sure on the dataset) instance of 
[Tensorflow Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) (used at inference time).

> [shakespeare.txt](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/shakespeare.txt)

This file is the dataset and composed of regular texts (see below what does it look like).

*First Citizen:*  
*Before we proceed any further, hear me speak.*
 
*All:*  
*Speak, speak.*

> [train.py](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/train.py)

Contains codes for training.

> [inference.py](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py)

Contains codes for inference.

# How to Train the Model
## A more depth look into [train.py](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/train.py) file
\
First, it gets the dataset from the specified url ([line 11](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L11)). Then reads the dataset to train the tokenizer object just mentioned above and trains the tokenizer ([line 18](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L18)). After training, encodes the dataset ([line 24](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L24)). Since this is a stateful model, all sequences in batch should be  start where the sequences at the same index number in the previous batch left off. Let's say a batch composes of 32 sequences. The 33th sequence (i.e. the first sequence in the second batch) should exactly start where the 1st sequence (i.e. first sequence in the first batch) ended up. The second sequence in the 2nd batch should start where 2nd sequnce in first batch ended up and so on. Codes between [line 28](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L28) and [line 48](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L48) do this and result the dataset. Codes between [line 53](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L53) and [line 57](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/6277815bfc43c361186238e00de060d160822e8b/train.py#L57) create the stateful model. Note that to be able to adjust <code>recurrent_dropout</code> hyperparameter you have to train the model on a GPU. After creation of model, a callback to reset states at the beginning of each epoch is created. Then the training start with the calling <code>fit</code> method and then model (see [tensorflow' entire model save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model)), model's weights and the tokenizer is saved.

# Usage of the Model
## Where the magic happens ([inference.py](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py) file) 
\
To be able use the model, it should first converted to a stateless model due to a stateful model expects a batch of inputs instead of just an input.
To do this a stateless model with the same architecture of [stateful model](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L44) should be created. Codes between [line 44](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L44) and [line 49](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L49) do this.
To load weights the model should be builded. After building weight are loaded to the stateless model. This model uses predicted character at time step *t* as an inputs at time *t + 1* to predict character at *t + 2* and this operation keep goes until the prediction of last character (in this case it 100 but you can change it whatever you want. Note that the longer sequences end up with more inaccurate results). To predict the next characters, first the provided initial character should be tokenized. [<code>preprocess</code>](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L7) function does this. To prevent repeated characters to be shown in the generated text, the next character should be selected from candidate characters randomly. The [<code>next_char</code>](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L11) function does this. The randomness can be controlled with <code>temperature</code> parameter (to learn usage of it check the comment at [line 30](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L30)). The [<code>complete_text</code>](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L18) function, takes a character as an argument, predicts the next character via [<code>next_char</code>](https://github.com/recep-yildirim/Fake-Shakespearean-Text-Generator/blob/master/inference.py#L11) function and concatenates the predicted character to the text. It repeats the process until to reach <code>n_chars</code>. Last, the stateless model will be saved also.

# Results
## Effects of the magic  
\
<code>print(complete_text("a"))</code>  

*arpet:*  
*like revenge borning and vinged him not.*

*lady good:*  
*then to know to creat it; his best,--lord*

--- 

<code>print(complete_text("k"))</code>

*ken countents.*  
*we are for free!*

*first man:*  
*his honour'd in the days ere in any since*  
*and all this ma*

---

<code>print(complete_text("f"))</code>

*ford*:  
*hold! we must percy and he was were good.*  

*gabes:*  
*by fair lord, my courters,*  
*sir.*

*nurse:*  
*well*

---
 
<code>print(complete_text("h"))</code>

*holdred?*  
*what she pass myself in some a queen*  
*and fair little heartom in this trumpet our hands?*  
*the*  
