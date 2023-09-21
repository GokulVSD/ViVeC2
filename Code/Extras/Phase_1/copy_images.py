import os
import shutil

if __name__ == '__main__':
    outputs_dir = os.path.join(os.getcwd(), "Outputs")
    report_dir = os.path.join(os.getcwd(), "Report", "Screenshots")
    for img_no in os.listdir(outputs_dir):
        img_dir = os.path.join(outputs_dir, img_no)
        for fd in os.listdir(img_dir):
            src_dir = os.path.join(img_dir, fd)
            if str(fd).endswith(".png"):
                dst = os.path.join(report_dir, os.path.basename(fd))
                shutil.copy(src_dir, dst)
            else:
                for img in os.listdir(src_dir):
                    dst_name = "Img_" + str(img_no) + "_" + fd + "_" + img
                    src = os.path.join(src_dir, img)
                    dst = os.path.join(report_dir, dst_name)
                    shutil.copy(src, dst)
