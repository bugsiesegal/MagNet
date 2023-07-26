import torch
from JAAEC import AmazingAutoEncoder

out = torch.load("/home/bugsie/PycharmProjects/MagNet/JAAEC_MagNet/3rri3e09/checkpoints/epoch=141-step=134843.ckpt")
out.keys()
model = AmazingAutoEncoder((1, 10_000, 8), (1, 16), 1e-6, num_layers=2, num_heads=2)
model.load_state_dict(out["state_dict"])
script = model.encoder.to_torchscript()
script = torch.jit.optimize_for_inference(script)
torch.jit.save(script, "encoder.pt")