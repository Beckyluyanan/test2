import gradio as gr
import BaseDeploy as bd

model_path = 'model/cls.onnx'

def predict(input_img):
    model = bd(model_path)
    result = model.inference(input_img)
    result = model.print_result(result)    
    return input_img,result


demo = gr.Interface(fn=predict,  inputs=gr.Image(shape=(28, 28),source="canvas"), outputs=["image","text"])
demo.launch(share=True)
