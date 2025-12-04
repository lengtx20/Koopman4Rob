import cv2
from mcap_data_loader.utils.hydra_utils import hydra_instance_from_config_path


transform = hydra_instance_from_config_path("configs/augmentation/image.yaml")

img_array = cv2.imread("/home/ghz/图片/摄像头/test.jpg")

# transform = v2.Compose(
#     [
#         v2.ToImage(),
#         # must set scale=True to get [0,1] range, otherwise the following transforms may not work as expected
#         v2.ToDtype(torch.float32, True),
#         v2.RandomPhotometricDistort(p=0.13),
#         v2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5)),
#         v2.RandomAdjustSharpness(sharpness_factor=2, p=1),
#         v2.RandomGrayscale(p=0.13),
#         v2.RandomErasing(p=0.13, scale=(0.02, 0.02), ratio=(0.3, 0.3), value="random"),
#     ]
# )
cv2.imshow("Original Image", img_array)
while True:
    for transformed_img_numpy in transform([img_array]):
        print(f"Transformed image shape: {transformed_img_numpy.shape}")
        cv2.imshow("Transformed Image", transformed_img_numpy)
        if cv2.waitKey(0) & 0xFF in [27, ord("q")]:  # Press 'Esc' or 'q' to
            cv2.destroyAllWindows()
            exit()
