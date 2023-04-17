import cv2 as cv

#accessing pretrained model
pretrained_model = cv.CascadeClassifier("face_detector.xml") 

#face detection with images
img = cv.imread("test.jpeg")


#convert img to grayscale, model only works on grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
coordinate_list = pretrained_model.detectMultiScale(gray, 1.1, 4) 
        
for (x,y,w,h) in coordinate_list:
    cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv.imshow("Face Detector", img)
cv.waitKey(0)



#face detection with webcam
capture = cv.VideoCapture(0) 
while True:
    boolean, frame = capture.read()
    if boolean == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #pass in grayscale image, scale factor, and minimum neighbours
        coordinate_list = pretrained_model.detectMultiScale(gray, 1.1, 3) 
        
        # drawing rectangle in frame
        for (x,y,w,h) in coordinate_list:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
        # Display detected face
        cv.imshow("Live Face Detection", frame)
        
        # condition to break out of while loop
        if cv.waitKey(20) == ord('x'):
            break
        
capture.release()
cv.destroyAllWindows()
            
