import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

import boto3
import cv2
import numpy as np
import panoramasdk

from uuid import uuid4
import socket
import json
import copy
from sympy import Point, Polygon

import zmq
from multiprocessing import Process

# Modify your account and other information here
REGION = "ap-southeast-1"
IOT_PUB_TOPIC = "panorama/people/event"
S3_BUCKET_NAME = "panorama-tailgating-img-tec"
DETECT_AREA = [(895, 903), (1327, 940), (1283, 1040), (842, 992)]

def plot_polygon(im, cordon_coordinates=None, color=[0, 0, 255], line_thickness=10):
    [cv2.line(im, cordon_coordinates[idx], cordon_coordinates[idx + 1], color, line_thickness) for idx in
     range(len(cordon_coordinates) - 1)]

    # Close the last point to 1st point
    cv2.line(im, cordon_coordinates[-1], cordon_coordinates[0], color, line_thickness)


class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""

        self.MODEL_NODE = "model_node"
        self.MODEL_DIM = 512
        self.frame_num = 0
        self.threshold = 50.
        # Desired class
        self.classids = [14.]

        self.s3_client = boto3.resource('s3', region_name=REGION)
        self.iot_client = boto3.client('iot-data', region_name=REGION)

        self.last_count = 0
        self.current_count = 0
        self.total_count = 0

        # Polygon settings
        p1, p2, p3, p4 = map(Point, DETECT_AREA)
        self.poly = Polygon(p1, p2, p3, p4)

        try:
            # Parameters
            logger.info('Getting parameters')
            self.threshold = self.inputs.threshold.get()
        except:
            logger.exception('Error during initialization.')
        finally:
            logger.info('Initialiation complete.')
            logger.info('Threshold: {}'.format(self.threshold))

        Process(target=self.listener).start()

        # Init zmq publisher
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")

    def upload_image(self, s3_url, copy_to_send):
        raw_serial = cv2.imencode('.png', copy_to_send)[1].tostring()
        fn = s3_url.split('/')[-1]
        bucket = s3_url.split('/')[-2]
        self.s3_client.Object(bucket, fn).put(Body=raw_serial, ContentType='image/PNG')

    def listener(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://localhost:5555")
        # Set setsockopt to receive all message
        socket.setsockopt(zmq.SUBSCRIBE, "".encode('utf-8'))

        while True:
            msg = socket.recv_pyobj()
            self.upload_image(*msg)
            #threading.Thread(target=self.ppe_iot_handler.detect_and_report, args=(*msg, )).start()

    def publish(self, socket, args):
        """Usage
        send_args = [[stream_ids[idx], result_boxes, self.cordon_area[idx], image_list[idx]], event_no]
        self.publish(socket, send_args)
        """
        logger.info(f'Publish to s3')
        socket.send_pyobj(args)

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        self.frame_num += 1
        logger.debug(self.frame_num)

        # Loop through attached video streams
        streams = self.inputs.video_in.get()
        for stream in streams:
            self.process_media(stream)

        self.outputs.video_out.put(streams)

    def process_media(self, stream):
        """Runs inference on a frame of video."""
        image_data = preprocess(stream.image, self.MODEL_DIM)
        logger.debug(image_data.shape)

        # Run inference
        inference_results = self.call({"data":image_data}, self.MODEL_NODE)

        # Process results (object deteciton)
        self.process_results(inference_results, stream)

    def process_results(self, inference_results, stream):
        """Processes output tensors from a computer vision model and annotates a video frame."""
        if inference_results is None:
            logger.warning("Inference results are None.")
            return

        num_people = 0

        class_data = None # Class Data
        bbox_data = None # Bounding Box Data
        conf_data = None # Confidence Data

        # Pulls data from the class holding the results
        # inference_results is a class, which can be iterated through
        # but inference_results has no index accessors (cannot do inference_results[0])

        k = 0
        now = datetime.now()

        for det_data in inference_results:
            if k == 0:
                class_data = det_data[0]
            if k == 1:
                conf_data = det_data[0]
            if k == 2:
                bbox_data = det_data[0]
                for a in range(len(conf_data)):
                    if conf_data[a][0] * 100 > self.threshold and class_data[a][0] in self.classids:
                        (left, top, right, bottom) = np.clip(det_data[0][a]/self.MODEL_DIM,0,1)
                        #stream.add_rect(left, top, right, bottom)
                        cv2.rectangle(stream.image, (int(left * 1920), int(top * 1080)), (int(right * 1920), int(bottom * 1080)), (255, 88, 9), 2)
                        # office settings
                        check_bottom = bottom * 1080
                        check_mid = ((left + right) / 2) * 1920

                        if self.poly.encloses_point(Point(check_mid, check_bottom)):
                            num_people += 1
                    else:
                        continue
            k += 1

        self.current_count = num_people

        send_args = None
        if self.current_count != self.last_count :
            logger.info('# people {}'.format(str(num_people)))
            if self.current_count > self.last_count :
                self.total_count += self.current_count - self.last_count

            ts = datetime.timestamp(now)
            data = {
                "timestamp": ts,
                "last_count": self.last_count,
                "current_count": self.current_count,
                "camera_name": stream.stream_id,
                "s3_url": "s3://{}/{}.png".format(S3_BUCKET_NAME, ts)
            }
            send_args = [data["s3_url"]]

            self.last_count = self.current_count
            message_json = json.dumps(data)
            self.iot_client.publish(
                topic=IOT_PUB_TOPIC,
                payload=bytes(message_json, "utf-8"),
                qos=1)

        cv2.putText(stream.image,
            text='# total people {}'.format(str(self.total_count)),
            org=(200, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_4)
        plot_polygon(stream.image, DETECT_AREA, line_thickness=3)

        if send_args:
            send_args.append(copy.deepcopy(stream.image))
            self.publish(self.socket, send_args)


def preprocess(img, size):
    """Resizes and normalizes a frame of video."""
    resized = cv2.resize(img, (size, size))
    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]  # RGB
    img = resized.astype(np.float32) / 255.  # converting array of ints to floats
    r, g, b = cv2.split(img)
    # normalizing per channel data:
    r = (r - mean[0]) / std[0]
    g = (g - mean[1]) / std[1]
    b = (b - mean[2]) / std[2]
    # putting the 3 channels back together:
    x1 = [[[], [], []]]
    x1[0][0] = r
    x1[0][1] = g
    x1[0][2] = b
    return np.asarray(x1)

def get_logger(name=__name__,level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=100000000, backupCount=2)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    try:
        logger.info("INITIALIZING APPLICATION")

        app = Application()
        logger.info("PROCESSING STREAMS")
        while True:
            app.process_streams()
    except:
        logger.exception('Exception during processing loop.')

logger = get_logger(level=logging.INFO)
main()
