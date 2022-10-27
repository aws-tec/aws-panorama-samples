from awscrt import mqtt
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

import command_line_utils;

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

        self.s3_client = boto3.resource('s3', region_name='ap-southeast-1')
        #self.HOST = "192.168.1.107"
        #self.PORT = 9527
        self.last_count = 0
        self.current_count = 0
        self.total_count = 0

        #self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.s3_thread =

        self.current_frame = None
        self.detect_area = [(895, 903), (1327, 940), (1283, 1040), (842, 992)]
        # office settings
        p1, p2, p3, p4 = map(Point, [(895, 903), (1327, 940), (1283, 1040), (842, 992)])
        self.poly = Polygon(p1, p2, p3, p4)
        # home settings
        # p1, p2, p3, p4, p5, p6 = map(Point, [(903, 382), (1085, 327), (1076, 503), (1156, 538), (1001, 626), (908, 574)])
        # self.poly = Polygon(p1, p2, p3, p4, p5, p6)

        """AWS IoT MQTT Configuration"""
        self.cmdUtils = command_line_utils.CommandLineUtils("PubSub - Send and recieve messages through an MQTT connection.")
        self.cmdUtils.add_common_mqtt_commands()
        self.cmdUtils.add_common_topic_message_commands()
        self.cmdUtils.add_common_proxy_commands()
        self.cmdUtils.add_common_logging_commands()
        self.cmdUtils.register_command("key", "<path>", "Path to your key in PEM format.", default="/opt/aws/panorama/storage/src/test.key", type=str)
        self.cmdUtils.register_command("cert", "<path>", "Path to your client certificate in PEM format.", default="/opt/aws/panorama/storage/src/test.crt", type=str)
        self.cmdUtils.register_command("ca_file", "<path>", "Path to your root CA in PEM format.", default="/opt/aws/panorama/storage/src/root.ca.pem", type=str)
        self.cmdUtils.register_command("endpoint", "<address_ep>", "Endpoint Name.", default="a1wantim6afpnp-ats.iot.ap-southeast-1.amazonaws.com", type=str)
        self.cmdUtils.register_command("port", "<int>", "Connection port. AWS IoT supports 443 and 8883 (optional, default=auto).", default=443, type=int)
        self.cmdUtils.register_command("client_id", "<str>", "Client ID to use for MQTT connection (optional, default='test-*').", default="panorama-" + str(uuid4()))
        self.cmdUtils.register_command("count", "<int>", "The number of messages to send (optional, default='10').", default=10, type=int)
        self.cmdUtils.register_command("is_ci", "<str>", "If present the sample will run in CI mode (optional, default='None')")

        self.cmdUtils.get_args()

        self.mqtt_connection = self.cmdUtils.build_mqtt_connection(self.on_connection_interrupted, self.on_connection_resumed)
        self.connect_future = self.mqtt_connection.connect()

        # Future.result() waits until a result is available
        self.connect_future.result()
        logger.info("AWS IoT: Connected!")

        # Subscribe
        #logger.info("Subscribing to topic '{}'...".format('panorama/badge/event'))
        #self.subscribe_future, self.packet_id = self.mqtt_connection.subscribe(
        #    topic='panorama/badge/event',
        #    qos=mqtt.QoS.AT_LEAST_ONCE,
        #    callback=self.on_message_received)

        #self.subscribe_result = self.subscribe_future.result()
        #logger.info("Subscribed with {}".format(str(self.subscribe_result['qos'])))

        try:
            # Parameters
            logger.info('Getting parameters')
            self.threshold = self.inputs.threshold.get()
        except:
            logger.exception('Error during initialization.')
        finally:
            logger.info('Initialiation complete.')
            logger.info('Threshold: {}'.format(self.threshold))

    def on_connection_interrupted(self, connection, error, **kwargs):
        logger.info("Connection interrupted. error: {}".format(error))

    def on_connection_resumed(self, connection, return_code, session_present, **kwargs):
        logger.info("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))

        if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
            logger.info("Session did not persist. Resubscribing to existing topics...")
            resubscribe_future, _ = connection.resubscribe_existing_topics()

            # Cannot synchronously wait for resubscribe result because we're on the connection's event-loop thread,
            # evaluate result with a callback instead.
            resubscribe_future.add_done_callback(self.on_resubscribe_complete)

    # Callback when the subscribed topic receives a message
    def on_message_received(self, topic, payload, dup, qos, retain, **kwargs):
        logger.info("Received message from topic '{}': {}".format(topic, payload))
        copy_to_send = copy.deepcopy(self.current_frame)
        raw_serial = cv2.imencode('.png', copy_to_send)[1].tostring()
        self.s3_client.Object('panorama-tailgating', 'test').put(Body=raw_serial, ContentType='image/PNG')

        #if received_count == cmdUtils.get_command("count"):
        #    received_all_event.set()

    def on_resubscribe_complete(self, resubscribe_future):
        resubscribe_results = resubscribe_future.result()
        logger.info("Resubscribe results: {}".format(resubscribe_results))

        for topic, qos in resubscribe_results['topics']:
            if qos is None:
                logger.info("Server rejected resubscribe to topic: {}".format(topic))

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

        self.current_frame = stream.image

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
                        stream.add_rect(left, top, right, bottom)
                        # office settings
                        check_bottom = bottom * 1080
                        check_mid = ((left + right) / 2) * 1920
                        # home settings
                        # check_bottom = bottom * 720
                        # check_mid = ((left + right) / 2) * 1280
                        if self.poly.encloses_point(Point(check_mid, check_bottom)):
                            num_people += 1
                    else:
                        continue
            k += 1

        self.current_count = num_people

        if self.current_count != self.last_count :
            logger.info('# people {}'.format(str(num_people)))
            if self.current_count > self.last_count :
                self.total_count += self.current_count - self.last_count
                # Pack image to upload to s3
                # 1666769959_298565.png

            data = {
                "timestamp": datetime.timestamp(now),
                "last_count": self.last_count,
                "current_count": self.current_count,
                "camera_name": stream.stream_id
            }
            self.last_count = self.current_count
            message_json = json.dumps(data)
            self.mqtt_connection.publish(
                topic='panorama/people/event',
                payload=message_json,
                qos=mqtt.QoS.AT_MOST_ONCE)

        #logger.info('# Tec {} {} {}'.format(str(num_people), str(self.current_conut), str(self.last_count)))
        stream.add_label('# total people {}'.format(str(self.total_count)), 0.1, 0.1)
        plot_polygon(stream.image, self.detect_area, line_thickness=3)


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
