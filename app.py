import gradio as gr
import BaseDeploy as bd

model_path = 'model/cls.onnx'

def predict(input_img):
    inverted_img = input_img
    for i in range(len(inverted_img)):
        for j in range(len(inverted_img[i])):
            for k in range(len(inverted_img[i][j])):
                inverted_img[i][j][k] = 255 - inverted_img[i][j][k]
    model = bd(model_path)
    result = model.inference(inverted_img)
    result = model.print_result(result)
    return inverted_img,result


demo = gr.Interface(fn=predict,  inputs=gr.Image(shape=(28, 28),source="canvas"), outputs=["image","text"])
demo.launch(share=True)
