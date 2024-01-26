from __future__ import print_function

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description="Retinaface")

parser.add_argument(
    "-m",
    "--trained_model",
    default="./weights/Resnet50_Final.pth",
    type=str,
    help="Trained state_dict file path to open",
)
parser.add_argument(
    "--network", default="resnet50", help="Backbone network mobile0.25 or resnet50"
)
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use cpu inference"
)
parser.add_argument(
    "--confidence_threshold", default=0.02, type=float, help="confidence_threshold"
)
parser.add_argument("--top_k", default=5000, type=int, help="top_k")
parser.add_argument("--nms_threshold", default=0.4, type=float, help="nms_threshold")
parser.add_argument("--keep_top_k", default=750, type=int, help="keep_top_k")
parser.add_argument(
    "-s",
    "--save_image",
    action="store_true",
    default=True,
    help="show detection results",
)
parser.add_argument(
    "--vis_thres", default=0.6, type=float, help="visualization_threshold"
)

parser.add_argument(
    "--output_folder", default="/home/khuy/Documents/main/face-recognition/datasets/", help="Foler to save deteted face"
)
parser.add_argument("--base_filename", default="user", help="Base filename for saved images")

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == "__main__":
    #khuy_edit_add_output_folder_and_begin_file_count
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    file_count = 0 

    delay_time = 2
    start_time = time.time()
    time_interval = 0.5  # 20 giây chia cho 50 ảnh
    last_saved_time = 0
    saved_images_count = 0

    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print("Finished loading model!")
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(gpu_name)
    net = net.to(device)

    resize = 1

    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(fps)

    # testing begin
    # for i in range(10):
    # image_path = "./curve/test.jpg"
    # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, img_raw = cam.read()
        if not ret:
            break

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print("net forward time: {:.4f}".format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # Avoid division by zero
            current_fps = frame_count / elapsed_time
            fps_text = "FPS: {:.2f}".format(current_fps)
            cv2.putText(img_raw, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate time camera
        current_time = time.time()
        if current_time - start_time > 12 + delay_time:
            break

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[: args.keep_top_k, :]
        landms = landms[: args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image and current_time - start_time > delay_time and current_time - last_saved_time >= time_interval and saved_images_count < 50:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue

                

                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(
                    img_raw,
                    text,
                    (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                )

                # landms
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            
                # Lưu ảnh khuôn mặt phát hiện được
                face = img_raw[b[1]:b[3], b[0]:b[2]]
                face_filename = f"{args.base_filename}_{saved_images_count}.jpg"
                face_filepath = os.path.join(args.output_folder, face_filename)
                cv2.imwrite(face_filepath, face)
                saved_images_count += 1  # Tăng số đếm sau mỗi lần lưu file
                last_saved_time = current_time

                if saved_images_count >= 10:
                    break

            # # save image
            # if args.save_image:
            #     folder_path = "/home/khuy/Documents/main/face-recognition/datasets/data/huy"
            #     if not os.path.exists(folder_path):
            #         os.makedirs(folder_path)

            #     name = os.path.join(folder_path, "huy.jpg")
            #     cv2.imwrite(name, img_raw)
            #     print(f"Image saved to: {name}")
        cv2.imshow('System Camera', img_raw)  # Hiển thị frame hiện tại

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break
    cam.release()  # Giải phóng camera
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ