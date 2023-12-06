import onnxruntime as rt
import time
import os
import sys
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse
import re
import multiprocessing
import platform

import cv2
import torchvision
from postprogress import *

# directory reach, 获取当前目录和父目录
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)# setting path，将父级目录路径添加到系统路径中，以供后续导入模块使用
from common_utils import *
from model_configs import *

# 编译基本选项
required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}

parser = argparse.ArgumentParser()  # 实例化一个参数解析器
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
parser.add_argument('-z','--run_model_zoo', action='store_true',  help='Run model zoo models')
args = parser.parse_args()  # 解析命令行参数
os.environ["TIDL_RT_PERFSTATS"] = "1"   # 设置环境变量 TIDL_RT_PERFSTATS 的值为 "1"

so = rt.SessionOptions()    # 创建一个会话选项对象

print("Available execution providers : ", rt.get_available_providers()) #可用的执行单元
#编译用图片
calib_images = ['../../../test_data/1.jpg',
'../../../test_data/2.jpg',
'../../../test_data/3.jpg',
'../../../test_data/4.jpg',
'../../../test_data/5.jpg',
]
#测试用图片
test_images =  ['../../../test_data/6.jpg',
'../../../test_data/7.jpg',
'../../../test_data/8.jpg',
'../../../test_data/9.jpg',
'../../../test_data/10.jpg',
] 

sem = multiprocessing.Semaphore(0)  # 创建
if platform.machine() == 'aarch64': #检查是否在板端
    ncpus = 1
else:
    ncpus = os.cpu_count()
idx = 0
nthreads = 0
run_count = 0

if "SOC" in os.environ: #检查是否设置了SOC环境变量，无则exit
    SOC = os.environ["SOC"]
else:
    print("Please export SOC var to proceed")
    exit(-1)
if (platform.machine() == 'aarch64'  and args.compile == True): #若在板端且需要编译，exit
    print("Compilation of models is only supported on x86 machine \n\
        Please do the compilation on PC and copy artifacts for running on TIDL devices " )
    exit(-1)
if(SOC == "am62"):
    args.disable_offload = True
    args.compile = False

#计算benchmark
def get_benchmark_output(interpreter):
    benchmark_dict = interpreter.get_TI_benchmark_data()    # 获取模型推理的统计数据字典
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if 'proc_start' in stat:
            value = stat.split("ts:subgraph_")
            value = value[1].split("_proc_start")
            subgraphIds.append(value[0])
    for i in range(len(subgraphIds)):        # 计算处理时间、拷贝输入时间和拷贝输出时间
        proc_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
        copy_time += cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    totaltime = benchmark_dict['ts:run_end'] -  benchmark_dict['ts:run_start']  #计算总时间
    return copy_time, proc_time, totaltime

#图像预处理并推理
def infer_image(sess, image_files, config):
    input_details = sess.get_inputs()
    input_name = input_details[0].name
    floating_model = (input_details[0].type == 'tensor(float)')   # 判断是否为浮点模型
    height = input_details[0].shape[2]  #384 / 256
    width  = input_details[0].shape[3]  #128
    print(image_files)
    #imgs=image_files.convert('RGB')
    imgs=image_files
    img_bgr = cv2.imread(imgs)
    print("image size:", img_bgr.shape)
    img_bgr2 = cv2.resize(img_bgr, ( width,height))
    print("image resize:", img_bgr2.shape)
    img_rgb = img_bgr2[:,:,::-1]    #(384/256, 128, 3)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 预处理-归一化
    input_tensor = img_rgb / 255    # 预处理-构造输入 Tensor
    input_tensor = np.expand_dims(input_tensor, axis=0) # 加 batch 维度 (1, 384, 128, 3)
    input_tensor = input_tensor.transpose((0, 3, 1, 2)) # N, C, H, W
    input_tensor = np.ascontiguousarray(input_tensor)   # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
    input_data = torch.from_numpy(input_tensor).to(device).float() # 转 Pytorch Tensor
    #input_data = input_data[:, :1, :, :]    #转单通道
    print(input_data.shape)

    #推理图片，计时
    start_time = time.time()  # 记录开始时间
    output = list(sess.run(None, {input_name: input_data.numpy()}))  # 进行推理并获取输出结果
    print("output.shape:", output[0].shape)
    stop_time = time.time()
    infer_time = stop_time - start_time  # 计算推理时间
    # 获取拷贝时间、子图处理时间和总时间
    copy_time, sub_graphs_proc_time, totaltime = get_benchmark_output(sess)
    proc_time = totaltime - copy_time

    return imgs, output, proc_time, sub_graphs_proc_time, height, width

#main 主程序####################################################################
def run_model(model, mIdx):
    print("\nRunning_Model : ", model, " \n")
    config = models_configs[model]
    # 将编译配置更新到 delegate_options 中
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)   
    #   拼接 "artifacts_folder" 的路径，将 model 名称添加到文件夹路径中
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/' 
    
    # delete the contents of this folder
    if args.compile or args.disable_offload:    # 如果命令行参数中有 --compile 或 --disable_offload
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    #编译和测试选不同的数据集
    if(args.compile == True):   # 如果参数中存在 --compile
        input_image = calib_images
        import onnx
        log = f'\nRunning shape inference on model {config["model_path"]} \n'
        print(log)
        onnx.shape_inference.infer_shapes_path(config['model_path'], config['model_path'])  # 根据校准图像执行形状推断
    else:
        input_image = test_images
    numFrames = config['num_images']
    if(args.compile):   # 如果 numFrames 大于校准帧数，则将其设置为校准帧数
        if numFrames > delegate_options['advanced_options:calibration_frames']:
            numFrames = delegate_options['advanced_options:calibration_frames']
    
    ############   set interpreter  ################################
    #根据不同的命令行参数选择不同的解释器
    if args.disable_offload : 
        EP_list = ['CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] , providers=EP_list,sess_options=so)
    elif args.compile:
        EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    else:
        EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    
    ############  run  session  ############################
    for i in range(len(input_image)):
        print("-----------image:", i, "-----------")
        input_images=input_image[i]
        # 运行推断函数，获取输出结果，处理时间和子图时间，以及高度和宽度
        imgs, output, proc_time, sub_graph_time, height, width  = infer_image(sess, input_images, config)
        # 计算总处理时间和子图时间
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
        total_proc_time = total_proc_time /1000000
        sub_graphs_time = sub_graphs_time/1000000

       # post processing enabled only for inference, 如果不是编译模式，则执行后处理
        if(args.compile == False):  
            output = deploy_preprocess(output[0])   #获取推理结果并进行处理
            #print(output)
            pred_points = get_predicted_points(output[0])   #得到预测点位
            print(pred_points)
            eval_results = {}
            eval_results['pred_points'] = pred_points
            img = cv2.imread(input_images)  #导入图片用来画线
            img_plot = plot_slots(img, eval_results)    #画线
            cv2.imshow('seed', img_plot)    #显示结果
            key = cv2.waitKey(500) & 0xFF
            cv2.destroyAllWindows()
            save_path = os.path.join('../../../output_images','test_image'+str(i+1)+'.jpg') #保存路径
            cv2.imencode('.jpg', img_plot)[1].tofile(save_path)
        
        if args.compile or args.disable_offload :   # 如果是编译模式或者禁用了offload，则生成参数YAML文件
            gen_param_yaml(delegate_options['artifacts_folder'], config, int(height), int(width))
        log = f'\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time/(i+1):10.2f}, Offload Time : {sub_graphs_time/(i+1):10.2f} , DDR RW MBs : 0\n \n ' #{classes} \n \n'
        print(log)  # 打印日志信息
        if ncpus > 1:   # 如果使用了多个CPU，则释放信号量
            sem.release()

models = ['seed']
log = f'\nRunning {len(models)} Models - {models}\n'
print(log)

#以下为线程控制，由此处进入运行程序>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def join_one(nthreads): # 定义一个函数来加入一个线程
    global run_count
    sem.acquire()     # 获取一个信号量，控制线程同步
    run_count = run_count + 1   # 增加运行计数
    return nthreads - 1 # 返回线程数减1

def spawn_one(models, idx, nthreads):   # 定义一个函数来创建并启动一个线程
     # 创建一个新的进程，目标函数是 run_model，参数是 models 和 idx
    p = multiprocessing.Process(target=run_model, args=(models,idx,))
    p.start()   # 启动进程
    return idx + 1, nthreads + 1    # 返回新的 idx 和 nthreads

if ncpus > 1:   # 如果有多个CPU，则创建并启动多个线程
    for t in range(min(len(models), ncpus)):
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    while idx < len(models):     # 当还有未处理的 model 时, 等待一个线程完成，并减少线程数
        nthreads = join_one(nthreads)
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    for n in range(nthreads):
        nthreads = join_one(nthreads)
else : #如果只有一个CPU：使用一个循环顺序地处理每个模型。每个模型会直接调用run_model函数进行处理。
    for mIdx, model in enumerate(models):
        run_model(model, mIdx)
