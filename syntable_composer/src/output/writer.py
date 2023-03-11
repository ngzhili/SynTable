import atexit
import numpy as np
import os
from PIL import Image
import queue
import sys
import threading


class DataWriter:
    """ For processing and writing output data to files. """

    def __init__(self, data_dir, num_worker_threads, max_queue_size=500):
        """ Construct DataWriter. """

        from omni.isaac.synthetic_utils import visualization

        self.visualization = visualization
        atexit.register(self.stop_threads)
        self.data_dir = data_dir

        # Threading for multiple scenes
        self.num_worker_threads = num_worker_threads

        # Initialize queue with a specified size
        self.q = queue.Queue(max_queue_size)
        self.threads = []

    def start_threads(self):
        """ Start worker threads. """

        for _ in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker, daemon=True)
            t.start()
            self.threads.append(t)

    def stop_threads(self):
        """ Waits for all tasks to be completed before stopping worker threads. """

        print("Finish writing data...")

        # Block until all tasks are done
        self.q.join()

        print("Done.")

    def worker(self):
        """ Processes task from queue. Each tasks contains groundtruth data and metadata which is used to transform the output and write it to disk. """

        while True:
            groundtruth = self.q.get()
            if groundtruth is None:
                break
            filename = groundtruth["METADATA"]["image_id"]
            viewport_name = groundtruth["METADATA"]["viewport_name"]
            for gt_type, data in groundtruth["DATA"].items():
                if gt_type == "RGB":
                    self.save_image(viewport_name, gt_type, data, filename)
                elif gt_type == "WIREFRAME":
                    self.save_image(viewport_name, gt_type, data, filename)
                elif gt_type == "DEPTH":
                    if groundtruth["METADATA"]["DEPTH"]["NPY"]:
                        self.save_PFM(viewport_name, gt_type, data, filename)
                    if groundtruth["METADATA"]["DEPTH"]["COLORIZE"]:
                        self.save_image(viewport_name, gt_type, data, filename)
                elif gt_type == "DISPARITY":
                    if groundtruth["METADATA"]["DISPARITY"]["NPY"]:
                        self.save_PFM(viewport_name, gt_type, data, filename)
                    if groundtruth["METADATA"]["DISPARITY"]["COLORIZE"]:
                        self.save_image(viewport_name, gt_type, data, filename)
                elif gt_type == "INSTANCE":
                    self.save_segmentation(
                        viewport_name,
                        gt_type,
                        data,
                        filename,
                        groundtruth["METADATA"]["INSTANCE"]["WIDTH"],
                        groundtruth["METADATA"]["INSTANCE"]["HEIGHT"],
                        groundtruth["METADATA"]["INSTANCE"]["COLORIZE"],
                        groundtruth["METADATA"]["INSTANCE"]["NPY"],
                    )
                elif gt_type == "SEMANTIC":
                    self.save_segmentation(
                        viewport_name,
                        gt_type,
                        data,
                        filename,
                        groundtruth["METADATA"]["SEMANTIC"]["WIDTH"],
                        groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"],
                        groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"],
                        groundtruth["METADATA"]["SEMANTIC"]["NPY"],
                    )
                elif gt_type in ["BBOX2DTIGHT", "BBOX2DLOOSE", "BBOX3D"]:
                    self.save_bbox(
                        viewport_name,
                        gt_type,
                        data,
                        filename,
                        groundtruth["METADATA"][gt_type]["COLORIZE"],
                        groundtruth["DATA"]["RGB"],
                        groundtruth["METADATA"][gt_type]["NPY"],
                    )
                elif gt_type == "CAMERA":
                    self.camera_folder = self.data_dir + "/" + str(viewport_name) + "/camera/"
                    np.save(self.camera_folder + filename + ".npy", data)
                elif gt_type == "POSES":
                    self.poses_folder = self.data_dir + "/" + str(viewport_name) + "/poses/"
                    np.save(self.poses_folder + filename + ".npy", data)
                else:
                    raise NotImplementedError
            self.q.task_done()

    def save_segmentation(
        self, viewport_name, data_type, data, filename, width=1280, height=720, display_rgb=True, save_npy=True
    ):
        """ Save segmentation mask data and visuals. """

        # Save ground truth data as 16-bit single channel png
        if save_npy:
            if data_type == "INSTANCE":
                data_folder = os.path.join(self.data_dir, viewport_name, "instance")
                data = np.array(data, dtype=np.uint8)
                img = Image.fromarray(data, mode="L")
            elif data_type == "SEMANTIC":
                data_folder = os.path.join(self.data_dir, viewport_name, "semantic")
                data = np.array(data, dtype=np.uint8)
                img = Image.fromarray(data, mode="L")

            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, filename + ".png")
            img.save(file, "PNG", bits=16)

        # Save ground truth data as visuals
        if display_rgb:
            image_data = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
            image_data += 1
            if data_type == "SEMANTIC":
                # Move close values apart to allow color values to separate more
                image_data = np.array((image_data * 17) % 256, dtype=np.uint8)
            color_image = self.visualization.colorize_segmentation(image_data, width, height, 3, None)
            color_image = color_image[:, :, :3]
            color_image_rgb = Image.fromarray(color_image, "RGB")

            if data_type == "INSTANCE":
                data_folder = os.path.join(self.data_dir, viewport_name, "instance", "visuals")
            elif data_type == "SEMANTIC":
                data_folder = os.path.join(self.data_dir, viewport_name, "semantic", "visuals")

            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, filename + ".png")
            color_image_rgb.save(file, "PNG")

    def save_image(self, viewport_name, img_type, image_data, filename):
        """ Save rgb data, depth visuals, and disparity visuals. """

        # Convert 1-channel groundtruth data to visualization image data
        def normalize_greyscale_image(image_data):
            image_data = np.reciprocal(image_data)
            image_data[image_data == 0.0] = 1e-5
            image_data = np.clip(image_data, 0, 255)
            image_data -= np.min(image_data)
            if np.max(image_data) > 0:
                image_data /= np.max(image_data)
            image_data *= 255
            image_data = image_data.astype(np.uint8)
            return image_data

        # Save image data as png
        if img_type == "RGB":
            data_folder = os.path.join(self.data_dir, viewport_name, "rgb")
            image_data = image_data[:, :, :3]
            img = Image.fromarray(image_data, "RGB")
        elif img_type == "WIREFRAME":
            data_folder = os.path.join(self.data_dir, viewport_name, "wireframe")
            image_data = np.average(image_data, axis=2)
            image_data = image_data.astype(np.uint8)
            img = Image.fromarray(image_data, "L")
        elif img_type == "DEPTH":
            image_data = image_data * 100
            image_data = normalize_greyscale_image(image_data)

            data_folder = os.path.join(self.data_dir, viewport_name, "depth", "visuals")
            img = Image.fromarray(image_data, mode="L")
        elif img_type == "DISPARITY":
            image_data = normalize_greyscale_image(image_data)

            data_folder = os.path.join(self.data_dir, viewport_name, "disparity", "visuals")
            img = Image.fromarray(image_data, mode="L")

        os.makedirs(data_folder, exist_ok=True)
        file = os.path.join(data_folder, filename + ".png")
        img.save(file, "PNG")

    def save_bbox(self, viewport_name, data_type, data, filename, display_rgb=True, rgb_data=None, save_npy=True):
        """ Save bbox data and visuals. """

        # Save ground truth data as npy
        if save_npy:
            if data_type == "BBOX2DTIGHT":
                data_folder = os.path.join(self.data_dir, viewport_name, "bbox_2d_tight")
            elif data_type == "BBOX2DLOOSE":
                data_folder = os.path.join(self.data_dir, viewport_name, "bbox_2d_loose")
            elif data_type == "BBOX3D":
                data_folder = os.path.join(self.data_dir, viewport_name, "bbox_3d")

            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, filename)
            np.save(file, data)

        # Save ground truth data and rgb data as visuals
        if display_rgb and rgb_data is not None:
            color_image = self.visualization.colorize_bboxes(data, rgb_data)
            color_image = color_image[:, :, :3]
            color_image_rgb = Image.fromarray(color_image, "RGB")

            if data_type == "BBOX2DTIGHT":
                data_folder = os.path.join(self.data_dir, viewport_name, "bbox_2d_tight", "visuals")
            if data_type == "BBOX2DLOOSE":
                data_folder = os.path.join(self.data_dir, viewport_name, "bbox_2d_loose", "visuals")
            if data_type == "BBOX3D":
                # 3D BBox visuals are not yet supported
                return

            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, filename + ".png")
            color_image_rgb.save(file, "PNG")

    def save_PFM(self, viewport_name, data_type, data, filename):
        """ Save Depth and Disparity data. """

        if data_type == "DEPTH":
            data_folder = os.path.join(self.data_dir, viewport_name, "depth")
        elif data_type == "DISPARITY":
            data_folder = os.path.join(self.data_dir, viewport_name, "disparity")

        os.makedirs(data_folder, exist_ok=True)
        file = os.path.join(data_folder, filename + ".pfm")
        self.write_PFM(file, data)

    def write_PFM(self, file, image, scale=1):
        """ Convert numpy matrix into PFM and save. """

        file = open(file, "wb")

        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write(b"PF\n" if color else b"Pf\n")
        file.write(b"%d %d\n" % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write(b"%f\n" % scale)

        image.tofile(file)
