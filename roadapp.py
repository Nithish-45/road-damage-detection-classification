import cv2
from flask import Flask,render_template,request
import ctypes
from keras.utils import load_img
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        img=request.files['img']
        img.save('static/road_img.jpg')

        def check(res):
            p2 = ["good", "poor", "satisfactory", "very_poor"]
            path = p2
            model = load_model('model455.h5', compile=False)
            # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            pred = model.predict(res)
            res = np.argmax(pred)
            res = path[res]
            print(res)
            return res

        def convert_img_to_tensor2(fpath):
            img = cv2.imread(fpath)
            img = cv2.resize(img, (256, 256))
            res = img_to_array(img)
            res = np.array(res, dtype=np.float16) / 255.0
            res = res.reshape(-1, 256, 256, 3)
            res = res.reshape(1, 256, 256, 3)
            return res

        t2 = "static/road_img.jpg"
        res = convert_img_to_tensor2(t2)
        msg1 = check(res)
        if msg1 == "very_poor":
            # Load the image
            image1 = cv2.imread(t2)
            # Convert the image to grayscale
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            # Apply GaussianBlur to remove noise
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # Detect edges using Canny edge detection
            edges = cv2.Canny(gray_blur, 30, 150)
            # Find contours in the edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the bounding rectangle coordinates for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Draw a rectangle around the largest pothole
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # drawing rectangle
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #         #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),

            #         image = cv2.imread(t2)
            user32 = ctypes.windll.user32
            screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            # Resize the image to half the size of the screen
            image = cv2.resize(image1, (int(screen_width / 2), int(screen_height / 2)))


            cv2.imwrite('static/road.jpg',image)



        elif msg1 == "poor":
            #        Load the image
            image = cv2.imread(t2)
            cv2.imwrite('static/road.jpg',image)



        elif msg1 == "satisfactory":
            # Load the image

            image1 = cv2.imread(t2)
            user32 = ctypes.windll.user32
            screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            image = cv2.resize(image1, (int(screen_width / 2), int(screen_height / 2)))

            cv2.imwrite('static/road.jpg',image)


        else:
            # Load the image
            image1 = cv2.imread(t2)
            user32 = ctypes.windll.user32
            screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            # Resize the image to half the size of the screen
            image = cv2.resize(image1, (int(screen_width / 2), int(screen_height / 2)))
            cv2.imwrite('static/road.jpg',image)


        return render_template('pothole_detection.html',res=msg1)
    else:
        return render_template('pothole_detection.html')


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")


