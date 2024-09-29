import rospy 
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import socket

class FrameResponderNode:
    def __init__(self):
        rospy.init_node('frame_responder_node', anonymous=True)
        self.bridge = CvBridge()
        self.socket_server = self.setup_socket_server('127.0.0.1', 8502)
        self.current_image = None
        self.subscriber = None

    def setup_socket_server(self, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.settimeout(None)
        server_socket.bind((host, port))
        server_socket.listen(1)
        return server_socket

    def image_callback(self, data):
        # Salvo l'immagine
        self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        rospy.loginfo("Image received, unsubscribing from topic.")
        self.subscriber.unregister()  # Unsubscribe dopo avere ricevuto limmagine

    def process_request(self, client_socket):
        # Subscribe to the topic solo quando ricevo la richiesta
        #self.subscriber = rospy.Subscriber("mono_object_detector_3d_result", Image, self.image_callback)
        self.subscriber = rospy.Subscriber("/camera_face/color/image_raw", Image, self.image_callback)

        # aspetto
        while self.current_image is None:
            rospy.sleep(0.1)

        # jpeg
        _, encoded_image = cv2.imencode('.jpg', self.current_image)
        # bytes
        image_bytes = encoded_image.tobytes()
        # mando la size e l'immagine su tcp
        image_size = len(image_bytes)
        client_socket.sendall(image_size.to_bytes(4, byteorder='big') + image_bytes)
        rospy.loginfo("Image sent to client.")
        self.current_image = None  # Reset

    def run(self):
        while not rospy.is_shutdown():
            client_socket, addr = self.socket_server.accept()
            with client_socket:
                rospy.loginfo(f"Received request from {addr}")
                self.process_request(client_socket)

if __name__ == '__main__':
    try:
        frame_responder_node = FrameResponderNode()
        frame_responder_node.run()
    except rospy.ROSInterruptException:
        pass

