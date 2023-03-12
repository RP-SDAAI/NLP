#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#install all the library before launching streamlit
#python -m pip install transformers 
#python -m pip install streamlit
#python -m pip install sentencepiece


# In[ ]:


import torch
import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline


# In[ ]:


st.title('Abstractive Summarization using BART and T5 transformer model')
st.markdown('Student Name: Lim Zhao Hong Student ID: 20065320')


# In[ ]:


st.caption('Bart model reference: https://huggingface.co/transformers/v3.0.2/model_doc/bart.html')
st.caption('T5 model reference: https://huggingface.co/transformers/v3.0.2/model_doc/t5.html')
model = st.selectbox('Select the model', ('BART', 'T5'))


# In[ ]:


if model == 'BART':
    _num_beams = 4
    _no_repeat_ngram_size = 3
    _length_penalty = 1
    _min_length = 12
    _max_length = 128
    _early_stopping = True
else:
    _num_beams = 4
    _no_repeat_ngram_size = 3
    _length_penalty = 2
    _min_length = 30
    _max_length = 200
    _early_stopping = True

col1, col2, col3 = st.columns(3)
_num_beams = col1.number_input("num_beams", value=_num_beams)
st.markdown('num_beams: the number of different possible sequences considered at each generation step. A larger value increases computation time but also increases the quality of the generated text')
_min_length = col1.number_input("min_length", value=_min_length)
st.markdown('min_length: The minimum number of tokens that an output text can have')
_max_length = col2.number_input("max_length", value=_max_length)
st.markdown('max_length: The maximum number of tokens that an output text can have')

text = st.text_area('Text Input')


# In[ ]:


def summary_model(input_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "BART":
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        input_text = str(input_text)
        input_text = ' '.join(input_text.split())
        input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
        summary_ids = bart_model.generate(input_tokenized,
                                          num_beams=_num_beams,
                                          min_length=_min_length,
                                          max_length=_max_length)

        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]              
       
        st.write('Summary')
        st.success(output[0])

    else:
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        input_text = str(input_text).replace('\n', '')
        input_text = ' '.join(input_text.split())
        input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
        summary_task = torch.tensor([[21603, 10]]).to(device)
        input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
        summary_ids = t5_model.generate(input_tokenized,
                                        num_beams=_num_beams,
                                        min_length=_min_length,
                                        max_length=_max_length)
        output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        st.write('Summary')
        st.success(output[0])


# In[ ]:


if st.button('Submit'):
    summary_model(text)

