import torch
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import streamlit as st
from streamlit import session_state as sn_state



paris = "Whether it’s because French is considered the “language of love” or because of the romantic walks along the Seine River, Paris has distinguished itself as the “City of Love.”"
newyork = "New York City is known by many nicknames—such as “the City that Never Sleeps” or “Gotham”—but the most popular one is probably “the Big Apple.”"
mumbai= "The opportunities here are endless, which is why Mumbai is often referred to as “the City of Dreams.”"

@st.experimental_singleton
def get_models(): 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return tokenizer, model

#st.sidebar.header('Input panel')
st.title("GPT-2 Text Generation")
option_selected = st.selectbox("Choose an example or write your own",("New York City - The Big Apple","Paris - City of Love","Mumbai - City of Dreams","Custom"))
if option_selected == "New York City - The Big Apple":
    sequence = st.text_area("Input Sequence", value = newyork, height=200)
elif option_selected == "Paris - City of Love":
    sequence = st.text_area("Input Sequence", value = paris, height=200)
elif option_selected == "Mumbai - City of Dreams":
    sequence = st.text_area("Input Sequence", value = mumbai, height=200)
elif option_selected == "Custom":
    sequence = st.text_area("Input Sequence", value = "", height=200)

if 'generate' not in sn_state or 'result' not in sn_state:
    sn_state['generate'] = False
    sn_state['result'] = ""

col1, col2, col3 = st.columns([1.5,1,5])

def gentext_click():
    tokenizer, model = get_models()
    sn_state['generate'] = True
    inputs = tokenizer.encode(sequence, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, do_sample=True,top_k=50)    
    sn_state['result'] = tokenizer.decode(outputs[0], skip_special_tokens=True)    

def clear_click():
    sn_state['generate'] = False
    sn_state['result'] = ""    

def genmore_click():
    tokenizer, model = get_models()
    sn_state['generate'] = False       
    inputs = tokenizer.encode(sn_state['result'], return_tensors='pt')
    outputs = model.generate(inputs, max_length=400, do_sample=True,top_k=50)
    sn_state['result'] = tokenizer.decode(outputs[0], skip_special_tokens=True)    

with col1: 
    #st.button('Generate Text', on_click=generate_text, args = (result))
    generate = st.button('Generate Text',on_click= gentext_click) 

with col2:
    #st.button('Clear', on_click=clear_text,kwargs = (result))
    clear = st.button('Clear',on_click= clear_click) 

with col3:
    if sn_state['generate']:
        more = st.button("Generate More",key = 'generatemore',disabled = False,on_click= genmore_click)
    else:
        more = st.button("Generate More",key = 'generatemore',disabled = True,on_click= genmore_click)  

if sn_state['result'] != "" :
    st.info(sn_state['result'])
else:
    st.write("")




