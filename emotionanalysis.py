from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('S__57819154_0.jpg')

try:
    obj = DeepFace.analyze(img_path = "S__57819154_0.jpg", actions = ['age', 'gender', 'race', 'emotion'])

    # Check if obj is a list or a dictionary
    if isinstance(obj, list):
        for result in obj:
            result_text = f'{result["age"]} years old {result["dominant_race"]} {result["dominant_emotion"]} {result["gender"]}'
            cv2.putText(img, result_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    elif isinstance(obj, dict):
        result_text = f'{obj["age"]} years old {obj["dominant_race"]} {obj["dominant_emotion"]} {obj["gender"]}'
        cv2.putText(img, result_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        print("Unexpected result:", obj)

    # Display the image with results
    imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

except Exception as e:
    print("An error occurred:", e)
