import torch
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import streamlit as st
from streamlit import session_state as sn_state

@st.experimental_singleton
def get_model_tokenizer(modelpath):    
    model = GPT2LMHeadModel.from_pretrained(modelpath)  
    tokenizer = GPT2Tokenizer.from_pretrained(modelpath)
    return model, tokenizer

st.write('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)
#st.write('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)
#st.title("GPT-2 Fine-Tuning with Harry Potter and the Sorcerer's Stone")
st.markdown("<h3>GPT-2 Fine-Tuning with Harry Potter and the Sorcerer's Stone</h3>", unsafe_allow_html=True)
sequence = st.sidebar.text_area("Enter a text", value = "", height=200)
if 'result' not in sn_state:
    sn_state['result'] = ""

def generate_messages(model,tokenizer,prompt_text,length,temperature = 0.7,k=20,p=0.9,repetition_penalty = 1.0):

    MAX_LENGTH = int(10000)
    def adjust_length_to_model(length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length
        
    length = adjust_length_to_model(length=length, max_sequence_length=model.config.max_position_embeddings)

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )

    if len(output_sequences.shape) > 2:
        print('in output')
        output_sequences.squeeze_()    

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):        
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        total_sequence = (prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :])
       
    return total_sequence

col1, col2 = st.sidebar.columns([1,1])

def gentext_click():
    modelpath = "E:\\Coursera\\Coursera\\MLRecap\\notes\\practice\\gpt2-hp\\test-clm\\test-clm"
    model, tokenizer = get_model_tokenizer(modelpath)
    temperature = 1.0
    k=400
    p=0.9
    repetition_penalty = 1.0
    length = 400    
    sn_state['result'] = generate_messages(model,tokenizer,sequence,length,temperature = temperature,k=k,p=p,repetition_penalty = repetition_penalty)

def clear_click():    
    sn_state['result'] = ""   

with col1: 
    #st.button('Generate Text', on_click=generate_text, args = (result))
    generate = st.sidebar.button('Generate Text',on_click= gentext_click) 

with col2:
    #st.button('Clear', on_click=clear_text,kwargs = (result))
    clear = st.sidebar.button('Clear',on_click= clear_click)  

if sn_state['result'] != "" :
    st.info(sn_state['result'])
else:
    st.write("")