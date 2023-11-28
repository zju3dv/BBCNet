import torch.onnx
from configs.train_options import TrainOptions
from model.BBCNet import DepthModel
import coremltools as ct

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
    model.load_state_dict({k.replace('module.','').replace('_orig_mod.',''):v for k,v in pretrained.items()},strict=True)
    print(model)
    return model

def main():
    print("starting")
    # model_path = "/mnt/cache/chuguanyi1/Workspace/dense_depth/logs/nyudepthv2/0906_test/epoch_16_model_big.ckpt"
    model_path = "/mnt/afs1/chuguanyi1/densedepth2/logs/lmdb_dataset/1123194129_test/epoch_25_model_big.ckpt"
    input = torch.randn(1, 3, 480, 640)
    # input1 = torch.randn([1, 1, 384, 512])
    model = load_torch_model(model_path)
    traced_model = torch.jit.trace(model,input)
    # mlmodel = ct.convert(traced_model,inputs=[ct.TensorType(shape=input.shape)],debug = True)
    # mlmodel = ct.convert(traced_model,inputs=[ct.TensorType(shape=input.shape)])
    # convert to mlmodel , with output name being "out_0"
    
    mlmodel = ct.convert(traced_model,inputs=[ct.TensorType(shape=input.shape)],outputs=[ct.TensorType(name="out_0")])
    mlmodel.save("CommonDepth1.mlmodel")
    print(mlmodel)

if __name__ == '__main__':
    main()
