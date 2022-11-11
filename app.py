import jetson.inference
import jetson.utils
from centroidtracker import CentroidTracker

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)
display = jetson.utils.videoOutput("display://0") 
camera = jetson.utils.videoSource("test.mp4")      

font = jetson.utils.cudaFont()
font = jetson.utils.cudaFont( size=32 )
ct = CentroidTracker(maxDisappeared=50, maxDistance=40)
object_id_list = []

while display.IsStreaming():
	img = camera.Capture()
	rects = []
	detections = net.Detect(img)
	for obj in detections:
		rects.append((int(obj.Left), int(obj.Bottom), int(obj.Right), int(obj.Top)))
	
	objects = ct.update(rects)
	for (objectId, bbox) in objects.items():
		if objectId not in object_id_list:
			object_id_list.append(objectId)
	
	font.OverlayText(img, img.width, img.height, "Count: {}".format(len(object_id_list)), 5, 5, (255, 0, 0), (0, 0, 0))
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))