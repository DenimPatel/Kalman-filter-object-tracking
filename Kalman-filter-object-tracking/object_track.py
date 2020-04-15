import numpy as np 
import cv2
from tracker import Tracker
import time
import imageio
images = [] # for GIF generation 

def createimage(w,h):
	"""
		generates a 3D matrix - w x h x 3 -> white background Image
	"""
	size = (w, h, 1)
	img = np.ones((w,h,3),np.uint8)*128
	return img

def main():

	# data coming from the npy file, Object detection part skipped
	data = np.array(np.load('Detections.npy'))[0:10,0:150,0:150]
	
	# initialize the tracking system
	tracker = Tracker(dist_threshold = 150, max_frame_skipped = 30, max_trace_length = 5)
	
	skip_frame_count = 0
	
	# define colors of the tracks for better visualization
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]

	# iterate for each measurement
	for i in range(data.shape[1]):
		centers = data[:,i,:]
		frame = createimage(512,512)
		if (len(centers) > 0):
			# performs DATA associaltion, Kalman filter propogation
			tracker.update(centers)

			# rest is just for visualization
			for j in range(len(tracker.objects_to_track)):
				if (len(tracker.objects_to_track[j].trace) > 1):
					x = int(tracker.objects_to_track[j].trace[-1][0,0])
					y = int(tracker.objects_to_track[j].trace[-1][0,1])
					tl = (x-10,y-10)
					br = (x+10,y+10)
					cv2.rectangle(frame,tl,br,track_colors[j],1)
					# show rectangle on the current position
					cv2.putText(frame,str(tracker.objects_to_track[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
					# show trace of the object
					for k in range(len(tracker.objects_to_track[j].trace)):
						x = int(tracker.objects_to_track[j].trace[k][0,0])
						y = int(tracker.objects_to_track[j].trace[k][0,1])
						cv2.circle(frame,(x,y), 3, track_colors[j],-1)
					cv2.circle(frame,(x,y), 6, track_colors[j],-1)
				cv2.circle(frame,(int(data[j,i,0]),int(data[j,i,1])), 6, (0,0,0),-1)
			cv2.imshow('image',frame)
			# for GIF generation
			cv2.imwrite("image"+str(i)+".jpg", frame)
			images.append(imageio.imread("image"+str(i)+".jpg"))
			
			time.sleep(0.2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	# for GIF generation
	imageio.mimsave('Multi-Object-Tracking.gif', images, duration=0.08)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
	
if __name__ == '__main__':
	main()