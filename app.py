from predict import Predictor, model_cfg  # 从predict模块导入Predictor类和model_cfg字典
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
import gradio as gr  # 导入gradio库，用于创建Web UI

# set a lot of global variables

predictor = None  # 初始化预测器为None
vocabulary = ["bat man, woman"]  # 初始化词汇表，包含示例词汇
input_image: Image.Image = None  # 初始化输入图像为None，类型注解为PIL.Image.Image
outputs: dict = None  # 初始化输出结果为None，类型注解为字典
cur_model_name: str = None  # 初始化当前模型名称为None，类型注解为字符串


def set_vocabulary(text):  # 定义设置词汇表的函数
    global vocabulary  # 声明vocabulary为全局变量
    vocabulary = text.split(",")  # 将输入的文本按逗号分割成词汇列表
    print("set vocabulary to", vocabulary)  # 打印设置后的词汇表


def set_input(image):  # 定义设置输入图像的函数
    global input_image  # 声明input_image为全局变量
    input_image = image  # 将输入的图像赋值给input_image
    print("set input image to", image)  # 打印设置后的输入图像信息


def set_predictor(model_name: str):  # 定义设置预测器的函数
    global cur_model_name  # 声明cur_model_name为全局变量
    if cur_model_name == model_name:  # 如果当前模型名称与输入模型名称相同
        return  # 则直接返回，无需重复设置
    global predictor  # 声明predictor为全局变量
    predictor = Predictor(**model_cfg[model_name])  # 使用模型配置初始化预测器
    print("set predictor to", model_name)  # 打印设置后的预测器名称
    cur_model_name = model_name  # 更新当前模型名称


set_predictor(list(model_cfg.keys())[0])  # 初始化预测器，使用model_cfg中的第一个模型


# for visualization
def visualize(vis_mode):  # 定义可视化函数
    if outputs is None:  # 如果输出结果为空
        return None  # 返回None
    return predictor.visualize(**outputs, mode=vis_mode)  # 调用预测器的visualize方法进行可视化


def segment_image(vis_mode, voc_mode, model_name):  # 定义图像分割函数
    set_predictor(model_name)  # 设置预测器
    if input_image is None:  # 如果输入图像为空
        return None  # 返回None
    global outputs  # 声明outputs为全局变量
    result = predictor.predict(  # 调用预测器的predict方法进行预测
        input_image, vocabulary=vocabulary, augment_vocabulary=voc_mode
    )
    outputs = result  # 将预测结果赋值给outputs

    return visualize(vis_mode)  # 返回可视化结果


def segment_e2e(image, vis_mode):  # 定义端到端的图像分割函数（未使用）
    set_input(image)  # 设置输入图像
    return segment_image(vis_mode)  # 调用segment_image函数进行分割和可视化


# gradio

with gr.Blocks(  # 创建gradio Blocks（应用的主要布局）
    css="""
               #submit {background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px;width: 20%;margin: 0 auto; display: block;}
 
                """  # 设置CSS样式
) as demo:  # 将Blocks对象赋值给demo变量
    gr.Markdown(  # 添加Markdown文本，显示应用标题
        f"<h1 style='text-align: center; margin-bottom: 1rem'>Side Adapter Network for Open-Vocabulary Semantic Segmentation</h1>"
    )
    gr.Markdown(  # 添加Markdown文本，显示论文链接
        """   
    This is the demo for our conference paper : "[Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2302.12242)".
    """
    )
    # gr.Image(type="pil", value="./resources/arch.png", shape=(460, 200), elem_id="arch") # 注释掉的图像组件
    gr.Markdown(  # 添加Markdown文本，显示分隔线
        """
        ---
        """
    )
    with gr.Row():  # 创建一个行布局
        image = gr.Image(type="pil", elem_id="input_image")  # 添加图像上传组件
        plt = gr.Image(type="pil", elem_id="output_image")  # 添加图像输出组件

    with gr.Row():  # 创建另一个行布局
        model_name = gr.Dropdown(  # 添加下拉选择框，用于选择模型
            list(model_cfg.keys()), label="Model", value="san_vit_b_16"
        )
        augment_vocabulary = gr.Dropdown(  # 添加下拉选择框，用于选择词汇扩展模式
            ["COCO-all", "COCO-stuff"],
            label="Vocabulary Expansion",
            value="COCO-all",
        )
        vis_mode = gr.Dropdown(  # 添加下拉选择框，用于选择可视化模式
            ["overlay", "mask"], label="Visualization Mode", value="overlay"
        )
    object_names = gr.Textbox(value=",".join(vocabulary), label="Object Names (Empty inputs will use the vocabulary specified in `Vocabulary Expansion`. Multiple names should be seperated with ,.)", lines=5)  # 添加文本框，用于输入目标物体名称

    button = gr.Button("Run", elem_id="submit")  # 添加运行按钮
    note = gr.Markdown(  # 添加Markdown文本，显示FAQ
        """
        ---
        ### FAQ
        - **Q**: What is the `Vocabulary Expansion` option for?
          **A**: The vocabulary expansion option is used to expand the vocabulary of the model. The model assign category to each area with `argmax`. When only a vocabulary with few thing classes is provided, it will produce much false postive.
        - **Q**: Error: `Unexpected token '<', " <h"... is not valid JSON.`. What should I do?
            **A**: It is caused by a timeout error. Possibly your image is too large for a CPU server. Please try to use a smaller image or run it locally on a GPU server.
        """
        )
    #

    object_names.change(set_vocabulary, [object_names], queue=False)  # 当文本框内容改变时，调用set_vocabulary函数
    image.change(set_input, [image], queue=False)  # 当上传的图像改变时，调用set_input函数
    vis_mode.change(visualize, [vis_mode], plt, queue=False)  # 当可视化模式改变时，调用visualize函数，并更新输出图像
    button.click(  # 当按钮被点击时
        segment_image, [vis_mode, augment_vocabulary, model_name], plt, queue=False  # 调用segment_image函数，并更新输出图像
    )
    demo.load(  # 当应用加载时
        segment_image, [vis_mode, augment_vocabulary, model_name], plt, queue=False  # 调用segment_image函数，并更新输出图像
    )

demo.queue().launch()  # 启动gradio应用，启用队列处理请求
