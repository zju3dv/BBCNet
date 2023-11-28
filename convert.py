import torch.onnx
from configs.train_options import TrainOptions
from model.BBCNet import DepthModel

def convert_torch_to_onnx(model, onnx_filename, input_val):
    device = torch.device('cpu')
    input = input_val.to(device)
    input_names = ["input1"]
    output_names = ["output"]
    torch.onnx.export(model, input, onnx_filename, verbose=True, input_names=input_names, output_names=output_names,opset_version=11)
    print("convert to ", onnx_filename, " done")

def load_torch_model(model_path):
    opt = TrainOptions()
    train_args = opt.initialize().parse_args()
    train_args.with_confidence = True
    train_args.export_onnx = True
    model = DepthModel(train_args)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    pretrained = torch.load(model_path,map_location=torch.device('cpu'))
    # model.load_state_dict({k.replace('module.',''):v for k,v in pretrained.items()},strict=True)
    model.load_state_dict({k.replace('module._orig_mod.',''):v for k,v in pretrained.items()},strict=True)

    print(model)
    return model

def main():
    print("starting")
    # model_path = "/mnt/cache/chuguanyi1/Workspace/dense_depth/logs/nyudepthv2/0906_test/epoch_16_model_big.ckpt"
    model_path = "/mnt/afs1/chuguanyi1/densedepth2/logs/lmdb_dataset/1123194129_test/epoch_25_model_big.ckpt"
    onnx_filename = "model.onnx"
    input = torch.randn([1, 3, 480, 640])
    # input1 = torch.randn([1, 1, 384, 512])
    model = load_torch_model(model_path)
    convert_torch_to_onnx(model, onnx_filename, input)

if __name__ == '__main__':
    main()
