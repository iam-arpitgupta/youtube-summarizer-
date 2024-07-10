import torch
import gradio

# Use a pipeline as a high-level helper
from transformers import pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = ("../models/models--sshleifer--distilbart-cnn-12-6/snapshots"
              "/a4f8f3ea906ed274767e9906dbaede7531d660ff")
text_summary = pipeline("summarization", model=model_path,
                torch_dtype=torch.bfloat16)



def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

#demo = gr.Interface(fn=summary,input="text",outputs="text")
demo =gr.Interface(fn=summary,
                   inputs=[gr.Textbox(label="input the text to summarize",lines=6)],
                   outputs=[gr.Textbox(label="summarized text",lines=4)],
                   title="@GenAI project : text summarize",
                   description="this applicaition will be used to summarize the text")
demo.launch()