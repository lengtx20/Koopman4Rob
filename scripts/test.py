from koopman4rob.data.mcap_data_utils import create_train_val_dataloader


model_path = "/home/ghz/blip2-itm-vit-g"
data_root = "/home/ghz/Work/OpenGHz/MCAP-DataLoader/data/example"
batch_size = 2
num_workers = 0
device = None

train_loader, val_loader = create_train_val_dataloader(
    model_path, data_root, batch_size, num_workers, device
)

for batch in train_loader:
    print(batch.shape)
    assert batch.shape[0] == batch_size
    assert batch.shape[1] == 7 + 256 + 7
    # for item in batch:
    #     print(item.shape)
    #     assert item.shape[0] == 7 + 256 + 7
    #     # print(item.keys())
    #     # for key, value in item.items():
    #     #     print(f"{key}: {value[0].shape}")
    break
